# app.py — Decisor de Rotina (versão final e corrigida)
import streamlit as st
import pandas as pd
import requests
import base64
from functools import lru_cache
from io import BytesIO
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------------
# CONFIGURAÇÕES DO GITHUB
# ---------------------------
REPO_OWNER = "dehgui"
REPO_NAME = "decisor-de-rotina"
FILE_PATH = "activities.csv"

RAW_CSV_URL = "https://raw.githubusercontent.com/dehgui/decisor-de-rotina/refs/heads/main/activities.csv"

GH_TOKEN = st.secrets.get("GH_TOKEN", None)

# ---------------------------
# PARÂMETROS DO MODELO
# ---------------------------
MAX_ENERGY = 5
MAX_HUNGER = 5
MAX_EMO = 5
DECAY_FACTOR = 0.7
WINDDOWN_PENALTY = 4.0
WINDDOWN_WINDOW = 2

# ---------------------------
# FUNÇÕES DO GITHUB
# ---------------------------
def get_file_sha():
    if GH_TOKEN is None:
        return None
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GH_TOKEN}"}
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        return resp.json().get("sha")
    return None


def update_github_csv(new_content, commit_message="Atualização automática pelo app"):
    """Atualiza activities.csv no GitHub."""
    if GH_TOKEN is None:
        return False

    sha = get_file_sha()
    if sha is None:
        return False

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GH_TOKEN}"}

    encoded = base64.b64encode(new_content.encode()).decode()

    data = {
        "message": commit_message,
        "content": encoded,
        "sha": sha
    }

    resp = requests.put(url, headers=headers, json=data)
    return resp.status_code in (200, 201)

# ---------------------------
# CARREGAR CSV DO GITHUB
# ---------------------------
@st.cache_data(ttl=60)
def load_activities_df():
    try:
        df = pd.read_csv(RAW_CSV_URL)
        required_cols = [
            "name", "cost", "delta_energy", "delta_hunger", "base_utility",
            "environment", "earliest_hour", "latest_hour", "winddown"
        ]
        for col in required_cols:
            if col not in df.columns:
                df[col] = ""
        return df[required_cols]
    except Exception as e:
        st.error(f"Erro ao carregar CSV: {e}")
        return pd.DataFrame(columns=required_cols)

# ---------------------------
# FUNÇÕES DO MDP
# ---------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def base_reward(act, state):
    hour, energy, hunger, emotional, money, env_pref = state

    try:
        base = float(act["base_utility"])
        cost = float(act["cost"])
        de = int(act["delta_energy"])
        dh = int(act["delta_hunger"])
        env = int(act["environment"])
    except:
        return -999

    r = base

    if energy <= 1 and de < 0:
        r -= 2.5

    if hunger >= 3 and dh >= 0:
        r -= 1.5

    if cost > money:
        r -= (cost - money) / 50 + 5.0

    if env_pref != 3 and env != 3 and env != env_pref:
        r -= 1.5

    return r


def transition(state, act):
    hour, energy, hunger, emotional, money, env_pref = state

    new_hour = hour + 1
    new_energy = clamp(energy + int(act["delta_energy"]), 0, MAX_ENERGY)
    new_hunger = clamp(hunger + int(act["delta_hunger"]), 0, MAX_HUNGER)
    new_money = max(0, money - float(act["cost"]))

    return (new_hour, new_energy, new_hunger, emotional, new_money, env_pref)


@lru_cache(maxsize=None)
def V(hour, energy, hunger, emotional, money, env_pref, sleep_hour, prev_idx, consec):
    if hour >= sleep_hour:
        return (0.0, [])

    best_val = -1e9
    best_seq = []

    for i, act in enumerate(ACTIVITIES_LIST):
        if hour < int(act["earliest_hour"]) or hour > int(act["latest_hour"]):
            continue

        if float(act["cost"]) > money:
            continue

        base_r = base_reward(act, (hour, energy, hunger, emotional, money, env_pref))

        act_wind = str(act["winddown"]).strip().lower() == "true"

        if i == prev_idx:
            reward = base_r * (DECAY_FACTOR ** consec)
        else:
            reward = base_r

        if hour >= sleep_hour - WINDDOWN_WINDOW and not act_wind:
            reward -= WINDDOWN_PENALTY

        new_state = transition((hour, energy, hunger, emotional, money, env_pref), act)
        next_consec = consec + 1 if i == prev_idx else 1

        future_val, future_seq = V(
            new_state[0], new_state[1], new_state[2], new_state[3],
            new_state[4], new_state[5], sleep_hour, i, next_consec
        )

        total = reward + future_val

        if total > best_val:
            best_val = total
            best_seq = [(hour, act["name"], reward, float(act["cost"]))] + future_seq

    return (best_val, best_seq)

# ---------------------------
# GRÁFICO
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
        if name in name_map:
            act = name_map[name]
            energy = clamp(energy + int(act["delta_energy"]), 0, MAX_ENERGY)
            hunger = clamp(hunger + int(act["delta_hunger"]), 0, MAX_HUNGER)
        else:
            energy = clamp(energy + 1, 0, MAX_ENERGY)

        hours.append(h + 1)
        energy_vals.append(energy)
        hunger_vals.append(hunger)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(hours, energy_vals, marker='o', label="Energia")
    ax.plot(hours, hunger_vals, marker='s', label="Fome")
    ax.set_xticks(range(start_hour, sleep_hour + 1))
    ax.legend()
    plt.tight_layout()
    return fig

# ---------------------------
# PDF EXPORT
# ---------------------------
def make_pdf_bytes(title, seq, total_util):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        buf = BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        w, h = A4

        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, h - 40, title)
        c.setFont("Helvetica", 11)
        c.drawString(40, h - 60, f"Utilidade total: {total_util:.2f}")

        y = h - 100
        for hora, nome, rec, custo in seq:
            c.drawString(40, y, f"{hora}:00 → {nome} (U={rec:.2f}, R${custo:.2f})")
            y -= 18
            if y < 60:
                c.showPage()
                y = h - 40

        c.save()
        buf.seek(0)
        return buf.read(), "application/pdf", "planejamento.pdf"

    except Exception:
        txt = ""
        for h, n, r, c in seq:
            txt += f"{h}:00 → {n} (U={r:.2f}, R${c:.2f})\n"
        return txt.encode(), "text/plain", "planejamento.txt"

# ---------------------------
# ESTILO / CSS
# ---------------------------
st.set_page_config(page_title="Decisor de Rotina", page_icon="🔮", layout="wide")

st.markdown("""
<style>
.reportview-container .main {
    background: #0f1724;
    color: white;
}
h1, h2, h3 {
    color: #ffd1e8;
}
.stButton>button {
    background-color: #ff6fa3 !important;
    color: white !important;
    border-radius: 8px;
}
.stDataFrame {
    background: rgba(255,255,255,0.06);
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# INTERFACE (TABS)
# ---------------------------
st.title("🔮 DECISOR DE ROTINA — Planejamento Diário")

tabs = st.tabs([
    "📅 Gerar planejamento",
    "🛠 Gerenciar atividades",
    "📋 Atividades cadastradas"
])

# ---------------------------
# TAB 1 — PLANEJAMENTO
# ---------------------------
with tabs[0]:

    st.header("Defina seu estado atual")

    col1, col2, col3 = st.columns(3)

    with col1:
        start_hour = st.number_input("Hora atual", 0, 23, 9)
        sleep_hour = st.number_input("Hora de dormir", 0, 23, 23)
        env_pref = st.selectbox("Ambiente preferido", (1, 2, 3),
                                format_func=lambda x: {1: "Dentro", 2: "Fora", 3: "Tanto faz"}[x])

    with col2:
        energy = st.slider("Energia", 0, 5, 3)
        hunger = st.slider("Fome", 0, 5, 2)

    with col3:
        emotional = st.slider("Estado emocional", 0, 5, 3)
        money = st.number_input("Dinheiro disponível (R$)", 0.0, 2000.0, 50.0)

    df = load_activities_df()
    ACTIVITIES_LIST = df.to_dict(orient="records")

    if st.button("✨ Gerar planejamento"):
        V.cache_clear()
        total, seq = V(start_hour, energy, hunger, emotional, money,
                       env_pref, sleep_hour, -1, 0)

        st.subheader(f"Utilidade total: {total:.2f}")

        for h, n, r, c in seq:
            st.write(f"**{h}:00** → {n} (utilidade {r:.2f}, custo R${c:.2f})")

        fig = plot_states(seq, start_hour, sleep_hour,
                          (start_hour, energy, hunger, emotional, money, env_pref))
        st.pyplot(fig)

        content, mime, filename = make_pdf_bytes("Planejamento Diário", seq, total)
        st.download_button("📥 Baixar planejamento", content, filename, mime)

# ---------------------------
# TAB 2 — GERENCIAR ATIVIDADES
# ---------------------------
with tabs[1]:

    st.header("Cadastrar nova atividade")

    name = st.text_input("Nome da atividade")
    cost = st.number_input("Custo (R$)", 0.0, 1000.0, 0.0)
    delta_energy = st.number_input("Δ Energia", -5, 5, 0)
    delta_hunger = st.number_input("Δ Fome", -5, 5, 0)
    base_utility = st.number_input("Utilidade base (1-10)", 1, 10, 5)

    environment = st.selectbox("Ambiente", (1, 2, 3),
                               format_func=lambda x: {1: "Dentro", 2: "Fora", 3: "Tanto faz"}[x])

    earliest_hour = st.number_input("Primeira hora possível", 0, 23, 6)
    latest_hour = st.number_input("Última hora possível", 0, 23, 22)

    winddown = st.radio("Adequada para final do dia?", (True, False))

    if st.button("Cadastrar atividade"):
        if not name.strip():
            st.warning("Digite um nome!")
        else:
            df2 = load_activities_df()
            df2.loc[len(df2)] = [
                name, cost, delta_energy, delta_hunger,
                base_utility, environment, earliest_hour, latest_hour, winddown
            ]
            ok = update_github_csv(df2.to_csv(index=False))
            if ok:
                st.success("Cadastrado com sucesso! 🎉")
                st.rerun()
            else:
                st.error("Falha ao atualizar GitHub. Verifique token/permissões.")

    st.markdown("---")

    st.header("Remover atividade")
    df3 = load_activities_df()

    if len(df3) > 0:
        choice = st.selectbox("Escolha", df3["name"].tolist())

        if st.button("Remover"):
            df_new = df3[df3["name"] != choice]
            ok = update_github_csv(df_new.to_csv(index=False))
            if ok:
                st.success("Removida com sucesso!")
                st.rerun()
            else:
                st.error("Erro ao atualizar GitHub.")
    else:
        st.info("Nenhuma atividade cadastrada.")

# ---------------------------
# TAB 3 — LISTAR ATIVIDADES
# ---------------------------
with tabs[2]:
    st.header("📋 Atividades cadastradas")
    df_all = load_activities_df()
    st.dataframe(df_all, use_container_width=True)
