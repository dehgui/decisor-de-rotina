import streamlit as st
import pandas as pd
import requests
import base64
import matplotlib.pyplot as plt
from functools import lru_cache
from io import BytesIO

# ================================================================
# CONFIGURAÇÃO BÁSICA DO APP
# ================================================================
st.set_page_config(page_title="Decisor de Rotina — Individual", page_icon="🔮", layout="wide")

REPO_OWNER = "dehgui"
REPO_NAME = "decisor-de-rotina"
GH_TOKEN = st.secrets.get("GH_TOKEN", None)

# ================================================================
# FUNÇÕES UTILITÁRIAS PARA ACESSO AO GITHUB VIA API
# ================================================================

def github_get_file_sha(path):
    """Retorna o SHA de um arquivo no GitHub (necessário para sobrescrever)."""
    if GH_TOKEN is None:
        return None
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{path}"
    headers = {"Authorization": f"token {GH_TOKEN}"}
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return r.json().get("sha")
    return None


def github_read_csv(path):
    """Lê um CSV armazenado no GitHub (via API, sem cache)."""
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{path}"
    headers = {"Authorization": f"token {GH_TOKEN}"}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return None

    content = r.json()["content"]
    decoded = base64.b64decode(content).decode()
    from io import StringIO
    return pd.read_csv(StringIO(decoded))


def github_write_csv(path, df, commit_message):
    """Escreve um CSV no GitHub, sobrescrevendo-o."""
    sha = github_get_file_sha(path)
    if sha is None and df is None:
        return False

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{path}"
    headers = {"Authorization": f"token {GH_TOKEN}"}

    csv_bytes = df.to_csv(index=False).encode()
    encoded = base64.b64encode(csv_bytes).decode()

    data = {
        "message": commit_message,
        "content": encoded,
        "sha": sha
    }

    resp = requests.put(url, headers=headers, json=data)
    return resp.status_code in (200, 201)


# ================================================================
# SISTEMA DE USUÁRIOS INDIVIDUAIS
# ================================================================

st.title("🔮 Decisor de Rotina — Versão Individual por Usuário")

st.markdown("""
Digite abaixo um **ID pessoal**, simples, sem espaço, como:

- `joao`
- `maria22`
- `andre`
- `profdaniel`

Cada usuário terá seu **próprio banco de atividades**, salvo automaticamente no GitHub.
""")

user_id = st.text_input("Seu ID pessoal:", placeholder="ex: andre")

if not user_id:
    st.stop()

FILE_PATH = f"data/{user_id}_activities.csv"

# Garante que o CSV do usuário existe
df_user = github_read_csv(FILE_PATH)

if df_user is None:
    df_user = pd.DataFrame(columns=[
        "name", "cost", "delta_energy", "delta_hunger",
        "base_utility", "environment", "earliest_hour",
        "latest_hour", "winddown"
    ])
    github_write_csv(FILE_PATH, df_user, f"Criar base do usuário {user_id}")

# ================================================================
# CARREGADOR FINAL QUE SEMPRE PEGA O CSV MAIS ATUAL
# ================================================================
def load_user_activities():
    df = github_read_csv(FILE_PATH)
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "name","cost","delta_energy","delta_hunger",
            "base_utility","environment","earliest_hour",
            "latest_hour","winddown"
        ])
    return df


# ================================================================
# MODELO DO MDP (idêntico, apenas usa df_user)
# ================================================================

MAX_ENERGY = 5
MAX_HUNGER = 5
DECAY_FACTOR = 0.7
WINDDOWN_PENALTY = 4.0
WINDDOWN_WINDOW = 2

def clamp(v, lo, hi): 
    return max(lo, min(hi, v))

def base_reward(act, state):
    hour, energy, hunger, emotional, money, env_pref = state
    base = float(act["base_utility"])
    cost = float(act["cost"])
    de = int(act["delta_energy"])
    dh = int(act["delta_hunger"])
    env = int(act["environment"])

    r = base
    if energy <= 1 and de < 0: r -= 2.5
    if hunger >= 3 and dh >= 0: r -= 1.5
    if cost > money: r -= (cost - money)/50 + 5
    if env_pref != 3 and env != 3 and env != env_pref: r -= 1.5
    return r

def transition(state, act):
    hour, energy, hunger, emo, money, env = state
    return (
        hour + 1,
        clamp(energy + int(act["delta_energy"]), 0, MAX_ENERGY),
        clamp(hunger + int(act["delta_hunger"]), 0, MAX_HUNGER),
        emo,
        max(0, money - float(act["cost"])),
        env
    )


@lru_cache(maxsize=None)
def V(hour, energy, hunger, emotional, money, env_pref,
      sleep_hour, prev_idx, consec,
      mand_idx_tuple, mand_rem_tuple,
      current_mand_pos, current_mand_consec):

    if hour >= sleep_hour:
        return (0.0, []) if all(x == 0 for x in mand_rem_tuple) else (-1e6, [])

    activities = ACTIVITIES
    remaining = sleep_hour - hour
    total_mand = sum(mand_rem_tuple)

    best_val, best_seq = -1e9, []

    # Bloco obrigatório sendo executado
    if current_mand_pos != -1:
        i = mand_idx_tuple[current_mand_pos]
        act = activities[i]
        hour_mod = hour % 24

        if not (int(act["earliest_hour"]) <= hour_mod <= int(act["latest_hour"])):
            return (-1e6, [])
        if float(act["cost"]) > money:
            return (-1e6, [])

        base_r = base_reward(act, (hour, energy, hunger, emotional, money, env_pref))
        r = base_r * (DECAY_FACTOR ** consec) if i == prev_idx else base_r
        if hour >= sleep_hour - WINDDOWN_WINDOW and not eval(str(act["winddown"])):
            r -= WINDDOWN_PENALTY

        mand_rem = list(mand_rem_tuple)
        mand_rem[current_mand_pos] -= 1

        next_pos = current_mand_pos if mand_rem[current_mand_pos] > 0 else -1
        next_consec = current_mand_consec + 1 if next_pos != -1 else 0

        nt = transition((hour, energy, hunger, emotional, money, env_pref), act)
        fv, fs = V(nt[0], nt[1], nt[2], nt[3], nt[4], nt[5],
                    sleep_hour, i, next_consec,
                    mand_idx_tuple, tuple(mand_rem),
                    next_pos, next_consec)

        return (r + fv, [(hour, act["name"], r, float(act["cost"]))] + fs)

    # Caso normal
    for i, act in enumerate(activities):
        hour_mod = hour % 24
        if not (int(act["earliest_hour"]) <= hour_mod <= int(act["latest_hour"])):
            continue
        if float(act["cost"]) > money:
            continue

        is_mand = i in mand_idx_tuple
        if total_mand > remaining:
            continue
        if total_mand == remaining and not is_mand:
            continue

        base_r = base_reward(act, (hour, energy, hunger, emotional, money, env_pref))
        r = base_r * (DECAY_FACTOR**consec) if i == prev_idx else base_r
        if hour >= sleep_hour - WINDDOWN_WINDOW and not eval(str(act["winddown"])):
            r -= WINDDOWN_PENALTY

        mand_rem = list(mand_rem_tuple)
        next_pos, next_consec = -1, 0

        if is_mand:
            idx_pos = mand_idx_tuple.index(i)
            mand_rem[idx_pos] -= 1
            if mand_rem[idx_pos] > 0:
                next_pos = idx_pos
                next_consec = 1

        nt = transition((hour, energy, hunger, emotional, money, env_pref), act)
        fv, fs = V(nt[0], nt[1], nt[2], nt[3], nt[4], nt[5],
                    sleep_hour, i, (consec+1 if i==prev_idx else 1),
                    mand_idx_tuple, tuple(mand_rem),
                    next_pos, next_consec)

        total = r + fv
        if total > best_val:
            best_val = total
            best_seq = [(hour, act["name"], r, float(act["cost"]))] + fs

    return (best_val, best_seq)

# ================================================================
# INTERFACE DO APP — TABS
# ================================================================

tabs = st.tabs(["📅 Gerar planejamento", "🛠 Gerenciar atividades", "📋 Minhas atividades"])

# ================================================================
# TAB 1 — GERAR PLANEJAMENTO
# ================================================================
with tabs[0]:

    df = load_user_activities()
    ACTIVITIES = df.to_dict(orient="records")

    st.header("Defina seu estado atual")

    col1, col2, col3 = st.columns(3)

    with col1:
        start_hour = st.number_input("Hora atual (0–23)", 0, 23, 9)
        sleep_hour = st.number_input("Hora de dormir (0–23)", 0, 23, 23)
        sleep_eff = sleep_hour + 24 if sleep_hour <= start_hour else sleep_hour
        env_pref = st.selectbox("Ambiente preferido",
                                (1,2,3),
                                format_func=lambda x: {1:"Dentro",2:"Fora",3:"Tanto faz"}[x])

    with col2:
        energy = st.slider("Energia (0–5)", 0, 5, 3)
        hunger = st.slider("Fome (0–5)", 0, 5, 2)

    with col3:
        emotional = st.slider("Estado emocional (0–5)", 0, 5, 3)
        money = st.number_input("Dinheiro disponível (R$)", 0.0, 2000.0, 50.0)

    st.subheader("Atividades obrigatórias (blocos contínuos)")
    mand_names = df["name"].tolist()
    selected_mand = st.multiselect("Escolher atividades obrigatórias", mand_names)

    mand_hours = {}
    for name in selected_mand:
        mand_hours[name] = st.number_input(f"Horas contínuas para {name}", 1, 24, 1)

    if st.button("✨ Gerar planejamento ótimo"):
        mand_idx, mand_rem = [], []

        for n, hrs in mand_hours.items():
            i = next((i for i,a in enumerate(ACTIVITIES) if a["name"]==n), None)
            if i is not None:
                mand_idx.append(i)
                mand_rem.append(hrs)

        V.cache_clear()
        total, seq = V(start_hour, energy, hunger, emotional, money, env_pref,
                       sleep_eff, -1, 0,
                       tuple(mand_idx), tuple(mand_rem),
                       -1, 0)

        if total < -1e5:
            st.error("Não foi possível montar o planejamento.")
        else:
            st.success(f"Utilidade total: {total:.2f}")
            for h, n, r, c in seq:
                st.write(f"{h}:00 → {n} (U={r:.2f}, R${c:.2f})")


# ================================================================
# TAB 2 — CADASTRAR/REMOVER ATIVIDADES
# ================================================================
with tabs[1]:

    st.header("Cadastrar nova atividade")

    name = st.text_input("Nome da atividade")
    cost = st.number_input("Custo (R$)", 0.0, 2000.0, 0.0)
    delta_energy = st.number_input("Variação de energia (-5 a 5)", -5, 5, 0)
    delta_hunger = st.number_input("Variação de fome (-5 a 5)", -5, 5, 0)
    base_utility = st.number_input("Utilidade base (1–10)", 1, 10, 5)
    environment = st.selectbox("Ambiente", (1,2,3))
    earliest_hour = st.number_input("Primeira hora possível", 0, 23, 6)
    latest_hour = st.number_input("Última hora possível", 0, 23, 22)
    winddown = st.radio("Boa para o fim do dia?", (True, False))

    if st.button("Cadastrar"):
        df = load_user_activities()
        df.loc[len(df)] = [
            name, cost, delta_energy, delta_hunger,
            base_utility, environment, earliest_hour,
            latest_hour, winddown
        ]
        ok = github_write_csv(FILE_PATH, df, f"{user_id}: adicionar atividade {name}")
        if ok:
            st.success("Atividade cadastrada!")
            st.rerun()
        else:
            st.error("Erro ao gravar no GitHub.")

    st.markdown("---")
    st.header("Remover atividade")

    df = load_user_activities()

    if df.empty:
        st.info("Nenhuma atividade cadastrada.")
    else:
        choice = st.selectbox("Escolher atividade para remover", df["name"])
        if st.button("Remover"):
            df = df[df["name"] != choice]
            ok = github_write_csv(FILE_PATH, df, f"{user_id}: remover atividade {choice}")
            if ok:
                st.success("Atividade removida!")
                st.rerun()
            else:
                st.error("Erro ao gravar no GitHub.")


# ================================================================
# TAB 3 — LISTAR
# ================================================================
with tabs[2]:

    st.header("Minhas atividades")

    df = load_user_activities()
    if df.empty:
        st.info("Nenhuma atividade cadastrada.")
    else:
        df_show = df.copy()
        df_show["environment"] = df_show["environment"].map({1:"Dentro",2:"Fora",3:"Tanto faz"})
        df_show["winddown"] = df_show["winddown"].apply(lambda x: "✔" if str(x).lower()=="true" else "")

        st.dataframe(df_show, use_container_width=True)
