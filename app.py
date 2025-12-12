# app.py — Decisor de Rotina (com obrigatórias contínuas e suporte à madrugada)
import streamlit as st
import pandas as pd
import requests
import base64
from functools import lru_cache
from io import BytesIO
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------------
# CONFIG GITHUB / CSV RAW
# ---------------------------
REPO_OWNER = "dehgui"
REPO_NAME = "decisor-de-rotina"
FILE_PATH = "activities.csv"
RAW_CSV_URL = f"https://raw.githubusercontent.com/dehgui/decisor-de-rotina/refs/heads/main/activities.csv"
GH_TOKEN = st.secrets.get("GH_TOKEN", None)

# ---------------------------
# MODELO / PARÂMETROS
# ---------------------------
MAX_ENERGY = 5
MAX_HUNGER = 5
DECAY_FACTOR = 0.7
WINDDOWN_PENALTY = 4.0
WINDDOWN_WINDOW = 2

# ---------------------------
# UTIL GITHUB
# ---------------------------
def get_file_sha():
    if GH_TOKEN is None:
        return None
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GH_TOKEN}"}
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return r.json().get("sha")
    return None

def update_github_csv(new_content, commit_message="Atualização automática pelo app"):
    if GH_TOKEN is None:
        return False
    sha = get_file_sha()
    if sha is None:
        return False
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GH_TOKEN}"}
    encoded = base64.b64encode(new_content.encode()).decode()
    payload = {"message": commit_message, "content": encoded, "sha": sha}
    resp = requests.put(url, headers=headers, json=payload)
    return resp.status_code in (200, 201)

# ---------------------------
# CARREGAR ACTIVITIES CSV
# ---------------------------
@st.cache_data(ttl=60)
def load_activities_df():
    try:
        df = pd.read_csv(RAW_CSV_URL)
    except Exception as e:
        st.error(f"Erro ao carregar activities.csv do GitHub: {e}")
        df = pd.DataFrame(columns=["name","cost","delta_energy","delta_hunger","base_utility","environment","earliest_hour","latest_hour","winddown"])
    # Garantir colunas
    expected = ["name","cost","delta_energy","delta_hunger","base_utility","environment","earliest_hour","latest_hour","winddown"]
    for c in expected:
        if c not in df.columns:
            df[c] = ""
    return df[expected]

# ---------------------------
# FUNÇÕES DO MDP
# ---------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def base_reward(act, state):
    hour, energy, hunger, emotional, money, env_pref = state
    try:
        base = float(act.get("base_utility", 0))
        cost = float(act.get("cost", 0))
        de = int(act.get("delta_energy", 0))
        dh = int(act.get("delta_hunger", 0))
        env = int(act.get("environment", 3))
    except:
        return -999
    r = base
    if energy <= 1 and de < 0:
        r -= 2.5
    if hunger >= 3 and dh >= 0:
        r -= 1.5
    if cost > money:
        r -= (cost - money)/50 + 5.0
    if env_pref != 3 and env != 3 and env != env_pref:
        r -= 1.5
    return r

def transition(state, act):
    hour, energy, hunger, emotional, money, env_pref = state
    new_hour = hour + 1
    new_energy = clamp(energy + int(act.get("delta_energy", 0)), 0, MAX_ENERGY)
    new_hunger = clamp(hunger + int(act.get("delta_hunger", 0)), 0, MAX_HUNGER)
    new_money = max(0.0, money - float(act.get("cost", 0)))
    return (new_hour, new_energy, new_hunger, emotional, new_money, env_pref)

# ---------------------------
# PROGRAMAÇÃO DINÂMICA (V)
# - mand_idx_tuple: tuple of activity indices (indexes into ACTIVITIES_LIST) that are mandatory
# - mand_remaining: tuple of remaining hours for each mandatory (aligned with mand_idx_tuple)
# - current_mand_pos: index into mand_idx_tuple indicating which mandatory block we're currently executing, or -1
# - current_mand_consec: how many consecutive hours already done in the current mandatory block
# ---------------------------
@lru_cache(maxsize=None)
def V(hour, energy, hunger, emotional, money, env_pref, sleep_hour, prev_idx, consec,
      mand_idx_tuple, mand_remaining_tuple, current_mand_pos, current_mand_consec):
    # Terminal
    if hour >= sleep_hour:
        # if obligations remain unfinished, impose huge penalty (so PD avoids)
        if any(x > 0 for x in mand_remaining_tuple):
            return (-1e6, [])
        return (0.0, [])

    activities = ACTIVITIES_LIST
    remaining_periods = sleep_hour - hour

    # total mandatory hours left
    total_mand_left = sum(mand_remaining_tuple)

    best_val = -1e9
    best_seq = []

    # If total_mand_left > remaining_periods: impossible state (shouldn't occur if we block appropriately)
    # Check whether we are currently inside a mandatory block
    if current_mand_pos != -1:
        # we must continue that mandatory activity until its remaining becomes 0
        mand_act_idx = mand_idx_tuple[current_mand_pos]
        # find action index in activities list equal mand_act_idx
        # iterate to find action with matching index
        i = mand_act_idx
        act = activities[i]
        # check time window and cost
        # determine hour mod 24 for window check (handles after-midnight hours)
        hour_mod = hour % 24
        earliest = int(act.get("earliest_hour", 0))
        latest = int(act.get("latest_hour", 23))
        if not (earliest <= hour_mod <= latest):
            # can't continue (invalid) -> return very negative
            return (-1e6, [])
        if float(act.get("cost", 0)) > money:
            return (-1e6, [])
        # compute reward with decay
        base_r = base_reward(act, (hour, energy, hunger, emotional, money, env_pref))
        if i == prev_idx:
            r = base_r * (DECAY_FACTOR ** consec)
        else:
            r = base_r
        # winddown penalty
        act_wind = str(act.get("winddown", "False")).strip().lower() == "true"
        if hour >= sleep_hour - WINDDOWN_WINDOW and not act_wind:
            r -= WINDDOWN_PENALTY

        # update mandatory remaining for this position
        mand_remaining = list(mand_remaining_tuple)
        mand_remaining[current_mand_pos] = max(0, mand_remaining[current_mand_pos] - 1)
        # if finished this mandatory block, set current_mand_pos -> -1
        next_current_pos = current_mand_pos if mand_remaining[current_mand_pos] > 0 else -1
        next_current_consec = current_mand_consec + 1 if next_current_pos == current_mand_pos else 0

        nt = transition((hour, energy, hunger, emotional, money, env_pref), act)
        future_val, future_seq = V(nt[0], nt[1], nt[2], nt[3], nt[4], nt[5], sleep_hour,
                                   i, next_current_consec,
                                   mand_idx_tuple, tuple(mand_remaining), next_current_pos, next_current_consec)
        total = r + future_val
        return (total, [(hour, act.get("name",""), r, float(act.get("cost",0)))] + future_seq)

    # Not currently inside a mandatory block
    for i, act in enumerate(activities):
        # time window check using hour modulus
        hour_mod = hour % 24
        try:
            earliest = int(act.get("earliest_hour", 0))
            latest = int(act.get("latest_hour", 23))
        except:
            earliest, latest = 0, 23
        if not (earliest <= hour_mod <= latest):
            continue
        # cost check
        if float(act.get("cost", 0)) > money:
            continue

        # If there are mandatory hours left, and remaining_periods equals total_mand_left,
        # then we must disallow any non-mandatory activity (to guarantee enough time).
        # Also if total_mand_left > remaining_periods => impossible; PD avoids.
        is_mand = False
        mand_pos_for_this = -1
        if mand_idx_tuple:
            for pos, mand_act_index in enumerate(mand_idx_tuple):
                if mand_act_index == i:
                    is_mand = True
                    mand_pos_for_this = pos
                    break

        if total_mand_left > remaining_periods:
            # impossible state, give big penalty
            continue

        if total_mand_left == remaining_periods and not is_mand:
            # must choose a mandatory now, otherwise cannot finish all mandatory hours
            continue

        # reward calculation
        base_r = base_reward(act, (hour, energy, hunger, emotional, money, env_pref))
        if i == prev_idx:
            r = base_r * (DECAY_FACTOR ** consec)
        else:
            r = base_r
        act_wind = str(act.get("winddown","False")).strip().lower() == "true"
        if hour >= sleep_hour - WINDDOWN_WINDOW and not act_wind:
            r -= WINDDOWN_PENALTY

        # If we choose a mandatory action, we'll start/continue its block and must make it continuous.
        mand_remaining = list(mand_remaining_tuple)
        next_current_pos = -1
        next_current_consec = 0
        if is_mand:
            # start/continue mandatory block for this mandatory pos
            # reduce remaining hours for that mandatory by 1
            mand_remaining[mand_pos_for_this] = max(0, mand_remaining[mand_pos_for_this] - 1)
            # set next_current_pos to this pos if still has remaining >0 (then force continuation next step)
            if mand_remaining[mand_pos_for_this] > 0:
                next_current_pos = mand_pos_for_this
                next_current_consec = 1
            else:
                next_current_pos = -1
                next_current_consec = 0

        nt = transition((hour, energy, hunger, emotional, money, env_pref), act)
        future_val, future_seq = V(nt[0], nt[1], nt[2], nt[3], nt[4], nt[5], sleep_hour,
                                   i, (consec+1 if i==prev_idx else 1),
                                   mand_idx_tuple, tuple(mand_remaining), next_current_pos, next_current_consec)
        total = r + future_val
        if total > best_val:
            best_val = total
            best_seq = [(hour, act.get("name",""), r, float(act.get("cost",0)))] + future_seq

    return (best_val, best_seq)

# ---------------------------
# PLOT: evolução energia/fome
# ---------------------------
def plot_states(seq, start_hour, sleep_hour, initial_state):
    hours = []
    energy_vals = []
    hunger_vals = []
    hour, energy, hunger, emo, money, env_pref = initial_state
    hours.append(hour)
    energy_vals.append(energy)
    hunger_vals.append(hunger)
    name_map = {a["name"]: a for a in ACTIVITIES_LIST}
    for (h, name, rec, cost) in seq:
        act = name_map.get(name)
        if act:
            energy = clamp(energy + int(act.get("delta_energy",0)), 0, MAX_ENERGY)
            hunger = clamp(hunger + int(act.get("delta_hunger",0)), 0, MAX_HUNGER)
        else:
            energy = clamp(energy + 1, 0, MAX_ENERGY)
        hours.append((h+1))
        energy_vals.append(energy)
        hunger_vals.append(hunger)
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(hours, energy_vals, marker='o', label='Energia')
    ax.plot(hours, hunger_vals, marker='s', label='Fome')
    ax.set_xlabel("Hora (pode exceder 24 se passar meia-noite)")
    ax.legend()
    plt.tight_layout()
    return fig

# ---------------------------
# PDF fallback
# ---------------------------
def make_pdf_bytes(title, seq, total_util):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        buf = BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        w, h = A4
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, h-40, title)
        c.setFont("Helvetica", 11)
        c.drawString(40, h-60, f"Utilidade total: {total_util:.2f}")
        y = h-100
        for hora, nome, rec, custo in seq:
            c.drawString(40, y, f"{hora}:00 -> {nome} (U={rec:.2f}, R${custo:.2f})")
            y -= 16
            if y < 80:
                c.showPage()
                y = h-40
        c.save()
        buf.seek(0)
        return buf.read(), "application/pdf", "planejamento.pdf"
    except Exception:
        txt = f"Utilidade total: {total_util:.2f}\n\n"
        for hora, nome, rec, custo in seq:
            txt += f"{hora}:00 -> {nome} (U={rec:.2f}, R${custo:.2f})\n"
        return txt.encode(), "text/plain", "planejamento.txt"

# ---------------------------
# UI / TABS
# ---------------------------
st.set_page_config(page_title="Decisor de Rotina", layout="wide", page_icon="🔮")

st.markdown("""
<style>
body { background: #0b1220; color: #ffffff; }
h1,h2,h3 { color: #ffd1e8; }
.stButton>button { background:#ff6fa3; color:white; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

st.title("🔮 DECISOR DE ROTINA — Planejamento Diário (MDP + PD)")

tabs = st.tabs(["📅 Gerar planejamento", "🛠 Gerenciar atividades", "📋 Atividades cadastradas"])

# ---------------------------
# TAB 1 — Gerar planejamento
# ---------------------------
with tabs[0]:
    st.header("Defina seu estado atual")
    col1, col2, col3 = st.columns(3)
    with col1:
        start_hour = st.number_input("Hora atual (0-23)", 0, 23, 9)
        sleep_hour = st.number_input("Hora de dormir (0-23) — se for após meia-noite, coloque hora (ex: 3)", 0, 23, 23)
        # fix sleep crossing midnight
        if sleep_hour <= start_hour:
            sleep_hour_effective = sleep_hour + 24
        else:
            sleep_hour_effective = sleep_hour
        env_pref = st.selectbox("Ambiente preferido", (1,2,3), format_func=lambda x: {1:"Dentro",2:"Fora",3:"Tanto faz"}[x])
    with col2:
        energy = st.slider("Energia (0-5)", 0, 5, 3)
        hunger = st.slider("Fome (0-5)", 0, 5, 2)
    with col3:
        emotional = st.slider("Estado emocional (0-5)", 0, 5, 3)
        money = st.number_input("Dinheiro disponível (R$)", 0.0, 2000.0, 50.0)

    st.markdown("---")
    st.subheader("Atividades obrigatórias para hoje (blocos contínuos)")
    df_act = load_activities_df()
    ACTIVITIES_LIST = df_act.to_dict(orient="records")

    # multiselect of activity names
    names = df_act["name"].tolist()
    mand_selected = st.multiselect("Selecione atividades obrigatórias (ordem não importa)", names)

    # for each selected, user sets minimum continuous hours
    mand_hours = {}
    for n in mand_selected:
        k = st.number_input(f"Horas contínuas mínimas para '{n}'", min_value=1, max_value=24, value=1, key=f"mand_{n}")
        mand_hours[n] = int(k)

    # button compute
    if st.button("✨ Gerar planejamento ótimo"):
        # prepare mandatory structures
        mand_idx = []
        mand_rem = []
        for name, hours in mand_hours.items():
            # find index in ACTIVITIES_LIST
            idx = next((i for i,a in enumerate(ACTIVITIES_LIST) if a["name"]==name), None)
            if idx is not None:
                mand_idx.append(idx)
                mand_rem.append(hours)
        mand_idx_tuple = tuple(mand_idx)
        mand_rem_tuple = tuple(mand_rem)

        # initial state includes effective sleep hour (handles midnight crossing)
        start = start_hour
        # call V with mand tuples etc.
        V.cache_clear()
        total, seq = V(start, energy, hunger, emotional, money, env_pref,
                       sleep_hour_effective, -1, 0,
                       mand_idx_tuple, mand_rem_tuple, -1, 0)
        if total < -1e5 or not seq:
            st.error("Não foi possível gerar um planejamento viável com as restrições atuais. Tente alterar as obrigatórias ou aumentar a janela.")
        else:
            st.subheader(f"Utilidade total: {total:.2f}")
            st.write("Planejamento sugerido:")
            for h, n, r, c in seq:
                st.write(f"- {h}:00 → {n} (U {r:.2f}, custo R${c:.2f})")
            fig = plot_states(seq, start, sleep_hour_effective, (start, energy, hunger, emotional, money, env_pref))
            st.pyplot(fig)
            content, mime, fname = make_pdf_bytes("Planejamento Diário", seq, total)
            st.download_button("📥 Baixar planejamento (PDF/TXT)", content, file_name=fname, mime=mime)

# ---------------------------
# TAB 2 — Gerenciar atividades (CRUD)
# ---------------------------
with tabs[1]:
    st.header("Cadastrar nova atividade")
    name = st.text_input("Nome da atividade")
    cost = st.number_input("Custo (R$)", 0.0, 2000.0, 0.0)
    delta_energy = st.number_input("Δ Energia (inteiro)", -5, 5, 0)
    delta_hunger = st.number_input("Δ Fome (inteiro)", -5, 5, 0)
    base_utility = st.number_input("Utilidade base (1-10)", 1, 10, 5)
    environment = st.selectbox("Ambiente", (1,2,3), format_func=lambda x: {1:"Dentro",2:"Fora",3:"Tanto faz"}[x])
    earliest_hour = st.number_input("Primeira hora possível (0-23)", 0, 23, 6)
    latest_hour = st.number_input("Última hora possível (0-23)", 0, 23, 22)
    winddown = st.radio("Adequada para o final do dia (winddown)?", (True, False), index=1)
    st.caption("Winddown: atividades tranquilas adequadas às últimas horas (ex.: meditar, ler, descansar).")

    if st.button("Cadastrar atividade"):
        if not name.strip():
            st.warning("Nome é obrigatório.")
        else:
            df = load_activities_df()
            df.loc[len(df)] = [name, cost, delta_energy, delta_hunger, base_utility, environment, earliest_hour, latest_hour, str(winddown)]
            ok = update_github_csv(df.to_csv(index=False), commit_message=f"Adicionar atividade: {name}")
            if ok:
                st.success("Atividade cadastrada e CSV atualizado no GitHub.")
                st.rerun()
            else:
                st.error("Falha ao atualizar o GitHub. Verifique GH_TOKEN/permissões.")

    st.markdown("---")
    st.header("Remover atividade")
    df_rem = load_activities_df()
    if not df_rem.empty:
        choice = st.selectbox("Escolha atividade para remover", df_rem["name"].tolist())
        if st.button("Remover"):
            new_df = df_rem[df_rem["name"] != choice]
            ok = update_github_csv(new_df.to_csv(index=False), commit_message=f"Remover atividade: {choice}")
            if ok:
                st.success("Atividade removida do GitHub.")
                st.rerun()
            else:
                st.error("Falha ao atualizar o GitHub.")
    else:
        st.info("Nenhuma atividade cadastrada.")

# ---------------------------
# TAB 3 — Atividades cadastradas
# ---------------------------
with tabs[2]:
    st.header("Atividades cadastradas (fonte: activities.csv no GitHub)")
    df_all = load_activities_df()
    if df_all.empty:
        st.info("Nenhuma atividade.")
    else:
        def env_label(v):
            try:
                v = int(v)
            except:
                v = 3
            return {1:"🏠 Dentro", 2:"🌳 Fora", 3:"💫 Tanto faz"}[v]
        df_show = df_all.copy()
        df_show["environment"] = df_show["environment"].apply(env_label)
        df_show["winddown"] = df_show["winddown"].apply(lambda x: "✅" if str(x).strip().lower()=="true" else " ")
        st.dataframe(df_show, use_container_width=True)
    st.caption("Edite diretamente no GitHub para mudanças manuais, ou use a aba Gerenciar atividades.")
