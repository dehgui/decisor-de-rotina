# ============================================================
# DECISOR DE ROTINA — Planejamento Diário (MDP + Programação Dinâmica)
# Esta versão inclui:
#  - Obrigatórias contínuas
#  - Suporte a atividades após a meia-noite
#  - CRUD completo com GitHub
#  - Cache inteligente + reset após updates
#  - Comentários apenas para funções importantes
# ============================================================

import streamlit as st
import pandas as pd
import requests
import base64
from functools import lru_cache
from io import BytesIO
import matplotlib.pyplot as plt

# ---------------------------
# CONFIGURAÇÕES DO REPOSITÓRIO
# ---------------------------
REPO_OWNER = "dehgui"
REPO_NAME = "decisor-de-rotina"
FILE_PATH = "activities.csv"
RAW_CSV_URL = f"https://raw.githubusercontent.com/dehgui/decisor-de-rotina/refs/heads/main/activities.csv"
GH_TOKEN = st.secrets.get("GH_TOKEN", None)

# ============================================================
# 🔧 Funções utilitárias de comunicação com o GitHub
# ============================================================

def get_file_sha():
    """
    Obtém o SHA atual do arquivo no GitHub.
    Necessário para sobrescrever arquivos via API.
    """
    if GH_TOKEN is None:
        return None

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GH_TOKEN}"}

    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return r.json().get("sha")
    return None


def update_github_csv(new_content, commit_message="Atualização automática"):
    """
    Envia uma atualização do CSV para o GitHub via API.
    """
    if GH_TOKEN is None:
        return False

    sha = get_file_sha()
    if sha is None:
        return False

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GH_TOKEN}"}

    encoded = base64.b64encode(new_content.encode()).decode()
    data = {"message": commit_message, "content": encoded, "sha": sha}

    resp = requests.put(url, headers=headers, json=data)
    return resp.status_code in (200, 201)


# ============================================================
# 🔄 Carregamento do CSV com CACHE
# ============================================================

@st.cache_data(ttl=1)
def load_activities_df():
    """
    Lê o arquivo activities.csv diretamente do RAW do GitHub.
    """
    df = pd.read_csv(RAW_CSV_URL)
    expected = ["name","cost","delta_energy","delta_hunger","base_utility","environment","earliest_hour","latest_hour","winddown"]
    return df[expected]


# ============================================================
# ⚙️ Funções principais do modelo (MDP)
# ============================================================

MAX_ENERGY = 5
MAX_HUNGER = 5
DECAY_FACTOR = 0.7
WINDDOWN_PENALTY = 4.0
WINDDOWN_WINDOW = 2


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def base_reward(act, state):
    """
    Cálculo da utilidade base da ação considerando:
    - Energia
    - Fome
    - Preferência de ambiente
    - Custo
    """
    hour, energy, hunger, emotional, money, env_pref = state

    try:
        base = float(act["base_utility"])
        cost = float(act["cost"])
        de = int(act["delta_energy"])
        dh = int(act["delta_hunger"])
        env = int(act["environment"])
    except:
        return -999

    reward = base

    if energy <= 1 and de < 0:
        reward -= 2.5

    if hunger >= 3 and dh >= 0:
        reward -= 1.5

    if cost > money:
        reward -= (cost - money) / 50 + 5

    if env_pref != 3 and env != 3 and env != env_pref:
        reward -= 1.5

    return reward


def transition(state, act):
    """
    Transição de estado: aplica os efeitos da atividade.
    """
    hour, energy, hunger, emotional, money, env_pref = state

    return (
        hour + 1,
        clamp(energy + int(act["delta_energy"]), 0, MAX_ENERGY),
        clamp(hunger + int(act["delta_hunger"]), 0, MAX_HUNGER),
        emotional,
        max(0, money - float(act["cost"])),
        env_pref
    )


# ============================================================
# 🧠 PROGRAMAÇÃO DINÂMICA (V)
# Avalia todas as sequências possíveis até a hora de dormir.
# Suporta:
#  - Atividades obrigatórias contínuas
#  - Penalização de repetição excessiva
#  - Penalização para não-winddown antes de dormir
# ============================================================

@lru_cache(maxsize=None)
def V(hour, energy, hunger, emotional, money, env_pref,
      sleep_hour, prev_idx, consec,
      mand_idx_tuple, mand_remaining_tuple,
      current_mand_pos, current_mand_consec):

    # Caso terminal
    if hour >= sleep_hour:
        if any(x > 0 for x in mand_remaining_tuple):
            return (-1e6, [])
        return (0, [])

    remaining_periods = sleep_hour - hour
    total_mandatory_left = sum(mand_remaining_tuple)
    best_val = -1e9
    best_seq = []

    # Se estivermos dentro de um bloco obrigatório contínuo:
    if current_mand_pos != -1:
        i = mand_idx_tuple[current_mand_pos]
        act = ACTIVITIES_LIST[i]
        hour_mod = hour % 24

        early = int(act["earliest_hour"])
        late = int(act["latest_hour"])
        if not (early <= hour_mod <= late):
            return (-1e6, [])

        if float(act["cost"]) > money:
            return (-1e6, [])

        base_r = base_reward(act, (hour, energy, hunger, emotional, money, env_pref))
        r = base_r * (DECAY_FACTOR ** consec) if i == prev_idx else base_r

        wind_ok = str(act["winddown"]).lower() == "true"
        if hour >= sleep_hour - WINDDOWN_WINDOW and not wind_ok:
            r -= WINDDOWN_PENALTY

        mand_remaining = list(mand_remaining_tuple)
        mand_remaining[current_mand_pos] -= 1

        next_pos = current_mand_pos if mand_remaining[current_mand_pos] > 0 else -1
        next_consec = current_mand_consec + 1 if next_pos != -1 else 0

        s2 = transition((hour, energy, hunger, emotional, money, env_pref), act)
        fv, seq = V(s2[0], s2[1], s2[2], s2[3], s2[4], s2[5],
                    sleep_hour, i, next_consec,
                    mand_idx_tuple, tuple(mand_remaining),
                    next_pos, next_consec)

        return (r + fv, [(hour, act["name"], r, float(act["cost"]))] + seq)

    # Se NÃO estivermos dentro de um bloco obrigatório:
    for i, act in enumerate(ACTIVITIES_LIST):
        hour_mod = hour % 24
        early = int(act["earliest_hour"])
        late = int(act["latest_hour"])
        if not (early <= hour_mod <= late):
            continue

        if float(act["cost"]) > money:
            continue

        is_mand = False
        mand_pos = -1

        for pos, idx in enumerate(mand_idx_tuple):
            if idx == i:
                is_mand = True
                mand_pos = pos
                break

        if total_mandatory_left > remaining_periods:
            continue

        if total_mandatory_left == remaining_periods and not is_mand:
            continue

        base_r = base_reward(act, (hour, energy, hunger, emotional, money, env_pref))
        r = base_r * (DECAY_FACTOR ** consec) if i == prev_idx else base_r

        wind_ok = str(act["winddown"]).lower() == "true"
        if hour >= sleep_hour - WINDDOWN_WINDOW and not wind_ok:
            r -= WINDDOWN_PENALTY

        mand_remaining = list(mand_remaining_tuple)
        next_pos = -1
        next_consec_mand = 0

        if is_mand:
            mand_remaining[mand_pos] -= 1
            if mand_remaining[mand_pos] > 0:
                next_pos = mand_pos
                next_consec_mand = 1

        s2 = transition((hour, energy, hunger, emotional, money, env_pref), act)
        fv, seq = V(s2[0], s2[1], s2[2], s2[3], s2[4], s2[5],
                    sleep_hour, i, (consec+1 if i==prev_idx else 1),
                    mand_idx_tuple, tuple(mand_remaining),
                    next_pos, next_consec_mand)

        if r + fv > best_val:
            best_val = r + fv
            best_seq = [(hour, act["name"], r, float(act["cost"]))] + seq

    return (best_val, best_seq)


# ============================================================
# 📊 Gráfico de evolução de energia e fome
# ============================================================

def plot_states(seq, start_hour, initial_state):
    hours = []
    E = []
    H = []

    hour, energy, hunger, emo, money, env_pref = initial_state

    hours.append(hour)
    E.append(energy)
    H.append(hunger)

    name_map = {a["name"]: a for a in ACTIVITIES_LIST}

    for h, name, r, cost in seq:
        act = name_map.get(name)
        if act:
            energy = clamp(energy + int(act["delta_energy"]), 0, MAX_ENERGY)
            hunger = clamp(hunger + int(act["delta_hunger"]), 0, MAX_HUNGER)

        hours.append(h+1)
        E.append(energy)
        H.append(hunger)

    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(hours, E, label="Energia")
    ax.plot(hours, H, label="Fome")
    ax.legend()
    return fig


# ============================================================
# 📄 Exportação para PDF ou TXT
# ============================================================

def make_pdf_bytes(title, seq, total):
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        buf = BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        w, h = A4
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, h-40, title)
        c.setFont("Helvetica", 10)
        c.drawString(40, h-60, f"Utilidade total: {total:.2f}")

        y = h - 100
        for hora, nome, r, custo in seq:
            c.drawString(40, y, f"{hora}:00 → {nome}  (U={r:.2f}, R${custo:.2f})")
            y -= 15
            if y < 60:
                c.showPage()
                y = h - 40

        c.save()
        buf.seek(0)
        return buf.read(), "application/pdf", "planejamento.pdf"

    except:
        txt = "\n".join(f"{h}:00 → {n} (U={r:.2f}, R${c:.2f})"
                        for h,n,r,c in seq)
        return txt.encode(), "text/plain", "planejamento.txt"


# ============================================================
# 🌟 INTERFACE PRINCIPAL
# ============================================================

st.set_page_config(page_title="Decisor de Rotina", layout="wide", page_icon="🔮")

st.title("🔮 DECISOR DE ROTINA — Planejamento Diário")

tabs = st.tabs(["📅 Gerar planejamento", "🛠 Gerenciar atividades", "📋 Atividades cadastradas"])


# ============================================================
# TAB 1 — GERAR PLANEJAMENTO
# ============================================================

with tabs[0]:
    st.header("Defina seu estado atual")

    col1, col2, col3 = st.columns(3)

    with col1:
        start_hour = st.number_input("Hora atual (0-23)", 0, 23, 9)
        sleep_hour = st.number_input("Hora de dormir (0-23)", 0, 23, 23)
        sleep_effective = sleep_hour + 24 if sleep_hour <= start_hour else sleep_hour

        env_pref = st.selectbox("Ambiente preferido",
                                (1,2,3),
                                format_func=lambda x: {1:"Dentro",2:"Fora",3:"Tanto faz"}[x])

    with col2:
        energy = st.slider("Energia (0-5)", 0, 5, 3)
        hunger = st.slider("Fome (0-5)", 0, 5, 2)

    with col3:
        emotional = st.slider("Estado emocional (0-5)", 0, 5, 3)
        money = st.number_input("Dinheiro disponível (R$)", 0.0, 2000.0, 50.0)

    st.markdown("---")

    st.subheader("Atividades obrigatórias para hoje (blocos contínuos)")
    df = load_activities_df()
    ACTIVITIES_LIST = df.to_dict(orient="records")

    names = df["name"].tolist()
    mand_select = st.multiselect("Selecione atividades obrigatórias", names)

    mand_hours = {}
    for n in mand_select:
        mand_hours[n] = st.number_input(
            f"Horas contínuas para '{n}'", 1, 24, 1, key=f"mand_{n}"
        )

    if st.button("✨ Gerar planejamento ótimo"):
        mand_idx = []
        mand_rem = []
        for name, hrs in mand_hours.items():
            idx = next((i for i,a in enumerate(ACTIVITIES_LIST) if a["name"]==name), None)
            if idx is not None:
                mand_idx.append(idx)
                mand_rem.append(hrs)

        mand_idx_t = tuple(mand_idx)
        mand_rem_t = tuple(mand_rem)

        V.cache_clear()

        total, seq = V(start_hour, energy, hunger, emotional, money, env_pref,
                       sleep_effective, -1, 0,
                       mand_idx_t, mand_rem_t, -1, 0)

        if total < -1e5 or not seq:
            st.error("Não foi possível gerar um planejamento viável.")
        else:
            st.subheader(f"Utilidade total: {total:.2f}")
            for h,n,r,c in seq:
                st.write(f"- {h}:00 → {n} (U={r:.2f}, R${c:.2f})")

            fig = plot_states(seq, start_hour, (start_hour, energy, hunger, emotional, money, env_pref))
            st.pyplot(fig)

            content, mime, fname = make_pdf_bytes("Planejamento Diário", seq, total)
            st.download_button("📥 Baixar planejamento", content, file_name=fname, mime=mime)


# ============================================================
# TAB 2 — GERENCIAR ATIVIDADES
# ============================================================

with tabs[1]:
    st.header("Cadastrar nova atividade")

    name = st.text_input("Nome da atividade")
    cost = st.number_input("Custo_
