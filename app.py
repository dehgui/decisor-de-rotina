import streamlit as st
import pandas as pd
from functools import lru_cache

# ======================================
# CONFIGURAÇÃO DO CSV (RAW)
# ======================================
GITHUB_CSV_URL = "https://raw.githubusercontent.com/dehgui/decisor-de-rotina/refs/heads/main/activities.csv"

MAX_ENERGY = 5
MAX_HUNGER = 5
MAX_EMO = 5

DECAY_FACTOR = 0.7        # decaimento quando repete atividade
WINDDOWN_PENALTY = 4.0    # penalidade para atividades inadequadas à noite
WINDDOWN_WINDOW = 2       # últimas 2h antes do sono


# ======================================
# CARREGAR ATIVIDADES DO GITHUB
# ======================================
def load_activities_raw(url):
    try:
        df = pd.read_csv(url)
        return df.to_dict(orient="records")
    except Exception as e:
        st.error(f"Erro ao carregar o CSV do GitHub: {e}")
        return []


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ======================================
# RECOMPENSA BASE
# ======================================
def base_reward(act, state):
    hour, energy, hunger, emotional, money, env_pref = state

    base = float(act["base_utility"])
    cost = float(act["cost"])
    de = int(act["delta_energy"])
    dh = int(act["delta_hunger"])
    env = int(act["environment"])

    r = base

    # penalização energia baixa + atividade que consome energia
    if energy <= 1 and de < 0:
        r -= 2.5

    # fome alta e atividade não ajuda
    if hunger >= 3 and dh >= 0:
        r -= 1.5

    # custo excede dinheiro
    if cost > money:
        r -= (cost - money) / 50 + 5.0

    # ambiente incompatível
    if env_pref != 3 and env != 3 and env != env_pref:
        r -= 1.5

    return r


# ======================================
# TRANSIÇÃO DE ESTADO (MDP)
# ======================================
def transition(state, act):
    hour, energy, hunger, emotional, money, env_pref = state

    new_hour = hour + 1
    new_energy = clamp(energy + int(act["delta_energy"]), 0, MAX_ENERGY)
    new_hunger = clamp(hunger + int(act["delta_hunger"]), 0, MAX_HUNGER)
    new_emotional = emotional
    new_money = max(0.0, money - float(act["cost"]))

    return (new_hour, new_energy, new_hunger, new_emotional, new_money, env_pref)


# ======================================
# PROGRAMAÇÃO DINÂMICA (BELLMAN)
# Agora com:
# ✔ Decaimento por repetição
# ✔ Identificação de winddown
# ======================================
@lru_cache(maxsize=None)
def V(hour, energy, hunger, emotional, money, env_pref, sleep_hour, prev_idx, consec):
    if hour >= sleep_hour:
        return (0.0, [])

    best_val = -1e9
    best_seq = []

    for i, act in enumerate(ACTIVITIES):

        # janela horária
        earliest = int(act["earliest_hour"])
        latest = int(act["latest_hour"])
        if not (earliest <= hour <= latest):
            continue

        # custo compatível
        if float(act["cost"]) > money:
            continue

        # recompensa base
        base_r = base_reward(act, (hour, energy, hunger, emotional, money, env_pref))

        # decaimento por repetição
        if i == prev_idx:
            r = base_r * (DECAY_FACTOR ** consec)
        else:
            r = base_r

        # conversão segura da coluna winddown
        act_wind = str(act.get("winddown", "False")).strip().lower() == "true"

        # penalização para atividades inadequadas perto do sono
        if hour >= sleep_hour - WINDDOWN_WINDOW and not act_wind:
            r -= WINDDOWN_PENALTY

        # transição de estado
        nt = transition((hour, energy, hunger, emotional, money, env_pref), act)

        # atualização do histórico de repetição
        next_prev = i
        next_consec = consec + 1 if i == prev_idx else 1

        # chamada recursiva PD
        future_val, future_seq = V(
            nt[0], nt[1], nt[2], nt[3], nt[4], nt[5],
            sleep_hour, next_prev, next_consec
        )

        total = r + future_val

        if total > best_val:
            best_val = total
            best_seq = [(hour, act["name"], r, float(act["cost"]))] + future_seq

    # fallback — descansar
    if best_val < -1e8:
        rest_r = 1.0
        nt = transition((hour, energy, hunger, emotional, money, env_pref), {
            "delta_energy": 1,
            "delta_hunger": 0,
            "cost": 0
        })
        future_val, future_seq = V(nt[0], nt[1], nt[2], nt[3], nt[4], nt[5], sleep_hour, -1, 0)
        return (rest_r + future_val, [(hour, "Descansar", rest_r, 0.0)] + future_seq)

    return (best_val, best_seq)


# ======================================
# STREAMLIT UI
# ======================================
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

st.write("### Banco de atividades (GitHub RAW)")
st.write(pd.DataFrame(ACTIVITIES))

if st.button("Gerar rotina ótima"):
    V.cache_clear()
    val, seq = V(start_hour, energy, hunger, emotional, money, env_pref, sleep_hour, -1, 0)

    st.write(f"## Utilidade total: {val:.2f}")
    st.write("### Rotina sugerida:")

    for hora, nome, rec, custo in seq:
        st.write(f"{hora}:00 → **{nome}** (utilidade {rec:.2f}, custo R${custo:.2f})")
