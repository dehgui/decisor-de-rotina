import streamlit as st
import pandas as pd
from functools import lru_cache

# ==============================
# CONFIGURAÇÃO DO CSV (RAW)
# ==============================
GITHUB_CSV_URL = "https://raw.githubusercontent.com/dehgui/decisor-de-rotina/refs/heads/main/activities.csv" 

MAX_ENERGY = 5
MAX_HUNGER = 5
MAX_EMO = 5

# ==============================
# CARREGAR ATIVIDADES DO GITHUB
# ==============================
def load_activities_raw(url):
    try:
        df = pd.read_csv(url)
        return df.to_dict(orient="records")
    except Exception as e:
        st.error(f"Erro ao carregar o CSV do GitHub: {e}")
        return []


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ==============================
# FUNÇÃO DE RECOMPENSA
# ==============================
def reward(act, state):
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


# ==============================
# TRANSIÇÃO DE ESTADO (MDP)
# ==============================
def transition(state, act):
    hour, energy, hunger, emotional, money, env_pref = state

    new_hour = hour + 1
    new_energy = clamp(energy + int(act["delta_energy"]), 0, MAX_ENERGY)
    new_hunger = clamp(hunger + int(act["delta_hunger"]), 0, MAX_HUNGER)
    new_emotional = emotional
    new_money = max(0.0, money - float(act["cost"]))

    return (new_hour, new_energy, new_hunger, new_emotional, new_money, env_pref)


# ==============================
# PROGRAMAÇÃO DINÂMICA (BELLMAN)
# ==============================
@lru_cache(maxsize=None)
def V(hour, energy, hunger, emotional, money, env_pref, sleep_hour):
    if hour >= sleep_hour:
        return (0.0, [])

    best_val = -1e9
    best_seq = []

    for act in ACTIVITIES:

        earliest = int(act["earliest_hour"])
        latest = int(act["latest_hour"])

        if not (earliest <= hour <= latest):
            continue

        if float(act["cost"]) > money:
            continue

        r = reward(act, (hour, energy, hunger, emotional, money, env_pref))
        nt = transition((hour, energy, hunger, emotional, money, env_pref), act)

        future_val, future_seq = V(*nt, sleep_hour)
        total = r + future_val

        if total > best_val:
            best_val = total
            best_seq = [(hour, act["name"], r, float(act["cost"]))] + future_seq

    if best_val < -1e8:
        rest = {
            "name": "Descansar",
            "delta_energy": +1,
            "delta_hunger": 0,
            "base_utility": 1,
            "cost": 0,
            "environment": 1,
            "earliest_hour": 0,
            "latest_hour": 23,
        }

        nt = transition((hour, energy, hunger, emotional, money, env_pref), rest)
        future_val, future_seq = V(*nt, sleep_hour)

        return (1.0 + future_val, [(hour, "Descansar", 1.0, 0.0)] + future_seq)

    return (best_val, best_seq)


# ==============================
# STREAMLIT UI
# ==============================
st.title("🔮 DECISOR DE ROTINA — MDP + Programação Dinâmica")

st.sidebar.header("Estado inicial")

start_hour = st.sidebar.number_input("Hora atual", 0, 23, 9)
sleep_hour = st.sidebar.number_input("Hora de dormir", 0, 23, 23)

energy = st.sidebar.slider("Energia", 0, 5, 3)
hunger = st.sidebar.slider("Fome", 0, 5, 2)
emotional = st.sidebar.slider("Estado emocional", 0, 5, 3)
money = st.sidebar.number_input("Dinheiro disponível", 0.0, 1000.0, 50.0)
env_pref = st.sidebar.radio("Ambiente preferido", [1, 2, 3],
    format_func=lambda x: {1: "Dentro", 2: "Fora", 3: "Tanto faz"}[x]
)

# Carregar atividades
ACTIVITIES = load_activities_raw(GITHUB_CSV_URL)

st.write("### Banco de atividades (direto do GitHub)")
st.write(pd.DataFrame(ACTIVITIES))

if st.button("Gerar rotina ótima"):
    V.cache_clear()
    val, seq = V(start_hour, energy, hunger, emotional, money, env_pref, sleep_hour)

    st.write(f"## Utilidade total: {val:.2f}")
    st.write("### Rotina sugerida:")

    for hora, nome, rec, custo in seq:
        st.write(f"{hora}:00 → **{nome}** (utilidade {rec:.2f}, custo R${custo:.2f})")
