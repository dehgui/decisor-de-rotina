import streamlit as st
import pandas as pd
import base64
import requests
from functools import lru_cache

# ====================================================
# CONFIGURAÇÃO DO GITHUB PARA LEITURA/ESCRITA DO CSV
# ====================================================
REPO_OWNER = "dehgui"        
REPO_NAME = "decisor-de-rotina" 
FILE_PATH = "activities.csv"     
RAW_CSV_URL = f"https://raw.githubusercontent.com/dehgui/decisor-de-rotina/refs/heads/main/activities.csv"

GH_TOKEN = st.secrets.get("GH_TOKEN", None)

# Função para pegar o SHA atual do arquivo (necessário para atualizar)
def get_file_sha():
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GH_TOKEN}"}
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return r.json()["sha"]
    else:
        st.error("Erro ao obter SHA do arquivo no GitHub.")
        return None

def update_github_csv(new_content):
    """Atualiza o CSV no GitHub via API."""
    sha = get_file_sha()
    if sha is None:
        return False

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GH_TOKEN}"}

    encoded = base64.b64encode(new_content.encode()).decode()

    data = {
        "message": "Atualização automática pelo Decisor de Rotina",
        "content": encoded,
        "sha": sha
    }

    response = requests.put(url, headers=headers, json=data)

    return response.status_code in [200, 201]


# ====================================================
# CARREGAR ATIVIDADES
# ====================================================
def load_activities():
    try:
        df = pd.read_csv(RAW_CSV_URL)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar o CSV: {e}")
        return pd.DataFrame()


# ====================================================
# LIMITADORES DO MODELO
# ====================================================
MAX_ENERGY = 5
MAX_HUNGER = 5
MAX_EMO = 5

DECAY_FACTOR = 0.7
WINDDOWN_PENALTY = 4.0
WINDDOWN_WINDOW = 2


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ====================================================
# FUNÇÃO DE RECOMPENSA BASE
# ====================================================
def base_reward(act, state):
    hour, energy, hunger, emotional, money, env_pref = state

    base = float(act["base_utility"])
    cost = float(act["cost"])
    de = int(act["delta_energy"])
    dh = int(act["delta_hunger"])
    env = int(act["environment"])

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


# ====================================================
# TRANSIÇÃO DE ESTADO DO MDP
# ====================================================
def transition(state, act):
    hour, energy, hunger, emotional, money, env_pref = state

    new_hour = hour + 1
    new_energy = clamp(energy + int(act["delta_energy"]), 0, MAX_ENERGY)
    new_hunger = clamp(hunger + int(act["delta_hunger"]), 0, MAX_HUNGER)
    new_emotional = emotional
    new_money = max(0.0, money - float(act["cost"]))

    return (new_hour, new_energy, new_hunger, new_emotional, new_money, env_pref)


# ====================================================
# PROGRAMAÇÃO DINÂMICA — MDP (Bellman)
# ====================================================
@lru_cache(maxsize=None)
def V(hour, energy, hunger, emotional, money, env_pref, sleep_hour, prev_idx, consec):
    if hour >= sleep_hour:
        return (0.0, [])

    best_val = -1e9
    best_seq = []

    for i, act in enumerate(ACTIVITIES_LIST):

        earliest = int(act["earliest_hour"])
        latest = int(act["latest_hour"])
        if not (earliest <= hour <= latest):
            continue

        if float(act["cost"]) > money:
            continue

        base_r = base_reward(act, (hour, energy, hunger, emotional, money, env_pref))

        # conversão segura winddown
        act_wind = str(act.get("winddown", "False")).strip().lower() == "true"

        # decaimento
        if i == prev_idx:
            r = base_r * (DECAY_FACTOR ** consec)
        else:
            r = base_r

        # penalização perto do sono
        if hour >= sleep_hour - WINDDOWN_WINDOW and not act_wind:
            r -= WINDDOWN_PENALTY

        nt = transition((hour, energy, hunger, emotional, money, env_pref), act)

        next_prev = i
        next_consec = consec + 1 if prev_idx == i else 1

        future_val, future_seq = V(
            nt[0], nt[1], nt[2], nt[3], nt[4], nt[5],
            sleep_hour, next_prev, next_consec
        )

        total = r + future_val

        if total > best_val:
            best_val = total
            best_seq = [(hour, act["name"], r, float(act["cost"]))] + future_seq

    return (best_val, best_seq)


# ====================================================
# INTERFACE — TABS
# ====================================================
st.title("🔮 DECISOR DE ROTINA — MDP + Programação Dinâmica")

tabs = st.tabs(["📅 Gerar rotina", "🛠 Gerenciar atividades"])


# ====================================================
# ABA 1 — GERAR ROTINA
# ====================================================
with tabs[0]:

    df = load_activities()
    st.write("### Banco de atividades atual")
    st.dataframe(df)

    st.sidebar.header("Estado inicial")

    start_hour = st.sidebar.number_input("Hora atual", 0, 23, 9)
    sleep_hour = st.sidebar.number_input("Hora de dormir", 0, 23, 23)

    energy = st.sidebar.slider("Energia", 0, 5, 3)
    hunger = st.sidebar.slider("Fome", 0, 5, 2)
    emotional = st.sidebar.slider("Estado emocional", 0, 5, 3)
    money = st.sidebar.number_input("Dinheiro disponível", 0.0, 2000.0, 50.0)
    env_pref = st.sidebar.radio(
        "Ambiente preferido",
        [1, 2, 3],
        format_func=lambda x: {1: "Dentro", 2: "Fora", 3: "Tanto faz"}[x]
    )

    ACTIVITIES_LIST = df.to_dict(orient="records")

    if st.button("Gerar rotina ótima"):
        V.cache_clear()
        val, seq = V(start_hour, energy, hunger, emotional, money, env_pref, sleep_hour, -1, 0)

        st.write(f"## Utilidade total: {val:.2f}")
        st.write("### Rotina sugerida:")

        for hora, nome, rec, custo in seq:
            st.write(f"{hora}:00 → **{nome}** (utilidade {rec:.2f}, custo R${custo:.2f})")


# ====================================================
# ABA 2 — GERENCIAR ATIVIDADES
# ====================================================
with tabs[1]:

    st.header("📌 Cadastrar nova atividade")

    name = st.text_input("Nome da atividade")
    cost = st.number_input("Custo (R$)", 0.0, 200.0, 0.0)
    delta_energy = st.number_input("Variação de energia (pode ser negativa)", -5, 5, 0)
    delta_hunger = st.number_input("Variação da fome (pode ser negativa)", -5, 5, 0)
    base_utility = st.number_input("Utilidade base (1 a 10)", 1, 10, 5)
    environment = st.radio("Ambiente", [1, 2, 3], format_func=lambda x: {
        1: "Dentro de casa",
        2: "Fora de casa",
        3: "Tanto faz"
    }[x])
    earliest_hour = st.number_input("Hora inicial possível", 0, 23, 6)
    latest_hour = st.number_input("Última hora possível", 0, 23, 22)
    winddown = st.radio(
        "É uma atividade adequada para o final do dia?",
        [True, False],
        format_func=lambda x: "Sim" if x else "Não"
    )

    if st.button("Cadastrar atividade"):
        new_df = load_activities()

        new_df.loc[len(new_df)] = [
            name, cost, delta_energy, delta_hunger, base_utility,
            environment, earliest_hour, latest_hour, winddown
        ]

        new_csv = new_df.to_csv(index=False)

        if update_github_csv(new_csv):
            st.success("Atividade cadastrada com sucesso! Atualize a página para ver.")
        else:
            st.error("Erro ao atualizar o CSV no GitHub.")

    st.markdown("---")

    st.header("🗑 Remover atividade")

    df_remove = load_activities()

    activity_to_remove = st.selectbox("Selecione a atividade para remover", df_remove["name"].tolist())

    if st.button("Remover atividade"):
        new_df = df_remove[df_remove["name"] != activity_to_remove]
        new_csv = new_df.to_csv(index=False)

        if update_github_csv(new_csv):
            st.success("Atividade removida com sucesso!")
        else:
            st.error("Erro ao atualizar o CSV no GitHub.")
