import streamlit as st
import pandas as pd
import requests
import base64
from functools import lru_cache
from io import BytesIO
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# CONFIGURAÇÃO DO GITHUB
# -------------------------------------------------------------------
REPO_OWNER = "dehgui"
REPO_NAME = "decisor-de-rotina"
FILE_PATH = "activities.csv"
RAW_CSV_URL = f"https://raw.githubusercontent.com/dehgui/decisor-de-rotina/refs/heads/main/activities.csv"
GH_TOKEN = st.secrets.get("GH_TOKEN", None)

# -------------------------------------------------------------------
# FUNÇÕES PARA OBTER E ATUALIZAR O CSV NO GITHUB
# -------------------------------------------------------------------
def get_file_sha():
    """Retorna o SHA atual do arquivo no GitHub. Usado para invalidar cache."""
    if GH_TOKEN is None:
        return None
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GH_TOKEN}"}
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return r.json().get("sha")
    return None


def update_github_csv(new_content, commit_message="Atualizacao automatica"):
    """Atualiza o arquivo no GitHub com novo conteúdo."""
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


# -------------------------------------------------------------------
# CARREGAMENTO DO CSV (CACHE CONTROLADO PELO SHA)
# -------------------------------------------------------------------
@st.cache_data
def load_activities_df(sha):
    """Lê o CSV do GitHub. O cache é invalidado quando o SHA muda."""
    df = pd.read_csv(RAW_CSV_URL)
    return df


# -------------------------------------------------------------------
# PARÂMETROS DO MODELO MDP
# -------------------------------------------------------------------
MAX_ENERGY = 5
MAX_HUNGER = 5
DECAY_FACTOR = 0.7
WINDDOWN_PENALTY = 4.0
WINDDOWN_WINDOW = 2


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def base_reward(act, state):
    """Calcula a utilidade base de realizar uma atividade em determinado estado."""
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
        r -= (cost - money)/50 + 5.0
    if env_pref != 3 and env != 3 and env != env_pref:
        r -= 1.5

    return r


def transition(state, act):
    """Aplica os efeitos da atividade no estado atual."""
    hour, energy, hunger, emotional, money, env_pref = state
    new_hour = hour + 1
    new_energy = clamp(energy + int(act["delta_energy"]), 0, MAX_ENERGY)
    new_hunger = clamp(hunger + int(act["delta_hunger"]), 0, MAX_HUNGER)
    new_money = max(0, money - float(act["cost"]))
    return (new_hour, new_energy, new_hunger, emotional, new_money, env_pref)


# -------------------------------------------------------------------
# FUNÇÃO PRINCIPAL DO MDP (V)
# -------------------------------------------------------------------
@lru_cache(maxsize=None)
def V(hour, energy, hunger, emotional, money, env_pref,
      sleep_hour, prev_idx, consec,
      mand_idx_tuple, mand_rem_tuple,
      current_mand_pos, current_mand_consec):

    if hour >= sleep_hour:
        if any(x > 0 for x in mand_rem_tuple):
            return (-1e6, [])
        return (0.0, [])

    remaining = sleep_hour - hour
    total_mand = sum(mand_rem_tuple)

    best_val = -1e9
    best_seq = []

    # ------------------------------------------------------------
    # SE ESTAMOS EM UM BLOCO OBRIGATÓRIO, A ATIVIDADE É FORÇADA
    # ------------------------------------------------------------
    if current_mand_pos != -1:
        i = mand_idx_tuple[current_mand_pos]
        act = ACTIVITIES_LIST[i]
        hour_mod = hour % 24

        eh = int(act["earliest_hour"])
        lh = int(act["latest_hour"])
        if not (eh <= hour_mod <= lh):
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
        fv, fs = V(nt[0], nt[1], nt[2], nt[3], nt[4], nt[5], sleep_hour,
                    i, next_consec,
                    mand_idx_tuple, tuple(mand_rem),
                    next_pos, next_consec)

        return (r + fv, [(hour, act["name"], r, float(act["cost"]))] + fs)

    # ------------------------------------------------------------
    # CASO NORMAL: PODEMOS ESCOLHER QUALQUER ATIVIDADE
    # ------------------------------------------------------------
    for i, act in enumerate(ACTIVITIES_LIST):
        hour_mod = hour % 24

        eh = int(act["earliest_hour"])
        lh = int(act["latest_hour"])
        if not (eh <= hour_mod <= lh):
            continue

        if float(act["cost"]) > money:
            continue

        is_mand = i in mand_idx_tuple

        if total_mand > remaining:
            continue

        if total_mand == remaining and not is_mand:
            continue

        base_r = base_reward(act, (hour, energy, hunger, emotional, money, env_pref))
        r = base_r * (DECAY_FACTOR ** consec) if i == prev_idx else base_r

        if hour >= sleep_hour - WINDDOWN_WINDOW and not eval(str(act["winddown"])):
            r -= WINDDOWN_PENALTY

        mand_rem = list(mand_rem_tuple)
        next_pos = -1
        next_consec = 0

        if is_mand:
            idx_pos = mand_idx_tuple.index(i)
            mand_rem[idx_pos] -= 1
            if mand_rem[idx_pos] > 0:
                next_pos = idx_pos
                next_consec = 1

        nt = transition((hour, energy, hunger, emotional, money, env_pref), act)
        fv, fs = V(nt[0], nt[1], nt[2], nt[3], nt[4], nt[5], sleep_hour,
                    i, (consec + 1 if i == prev_idx else 1),
                    mand_idx_tuple, tuple(mand_rem),
                    next_pos, next_consec)

        total = r + fv
        if total > best_val:
            best_val = total
            best_seq = [(hour, act["name"], r, float(act["cost"]))] + fs

    return (best_val, best_seq)


# -------------------------------------------------------------------
# PLOT DE EVOLUÇÃO DE ESTADOS
# -------------------------------------------------------------------
def plot_states(seq, start_hour, initial_state):
    hours = []
    ev = []
    hv = []

    hour, energy, hunger, emo, money, env_pref = initial_state
    hours.append(hour)
    ev.append(energy)
    hv.append(hunger)

    name_map = {a["name"]: a for a in ACTIVITIES_LIST}

    for h, name, _, _ in seq:
        act = name_map[name]
        energy = clamp(energy + int(act["delta_energy"]), 0, MAX_ENERGY)
        hunger = clamp(hunger + int(act["delta_hunger"]), 0, MAX_HUNGER)

        hours.append(h+1)
        ev.append(energy)
        hv.append(hunger)

    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(hours, ev, marker="o", label="Energia")
    ax.plot(hours, hv, marker="s", label="Fome")
    ax.set_xlabel("Hora")
    ax.legend()
    return fig


# -------------------------------------------------------------------
# INTERFACE STREAMLIT
# -------------------------------------------------------------------
st.set_page_config(page_title="Decisor de Rotina", layout="wide")

st.title("🔮 Decisor de Rotina – Planejamento Diário")

tabs = st.tabs(["📅 Gerar planejamento", "🛠 Gerenciar atividades", "📋 Atividades cadastradas"])

# -------------------------------------------------------------------
# ABA 1 - GERAR PLANEJAMENTO
# -------------------------------------------------------------------
with tabs[0]:

    sha = get_file_sha()
    df_act = load_activities_df(sha)
    global ACTIVITIES_LIST
    ACTIVITIES_LIST = df_act.to_dict(orient="records")

    st.header("Estado atual")

    col1, col2, col3 = st.columns(3)

    with col1:
        start_hour = st.number_input("Hora atual (0-23)", 0, 23, 9)
        sleep_hour = st.number_input("Hora de dormir (0-23)", 0, 23, 23)
        sleep_eff = sleep_hour + 24 if sleep_hour <= start_hour else sleep_hour
        env_pref = st.selectbox("Ambiente preferido", (1,2,3),
                                format_func=lambda x: {1:"Dentro",2:"Fora",3:"Tanto faz"}[x])

    with col2:
        energy = st.slider("Energia (0-5)", 0, 5, 3)
        hunger = st.slider("Fome (0-5)", 0, 5, 2)

    with col3:
        emotional = st.slider("Estado emocional (0-5)", 0, 5, 3)
        money = st.number_input("Dinheiro disponível (R$)", 0.0, 2000.0, 50.0)

    st.subheader("Atividades obrigatórias (blocos contínuos)")
    names = df_act["name"].tolist()
    mand_selected = st.multiselect("Selecione atividades", names)

    mand_hours = {n: st.number_input(f"Horas mínimas contínuas para {n}", 1, 24, 1)
                  for n in mand_selected}

    if st.button("✨ Gerar planejamento ótimo"):
        mand_idx = []
        mand_rem = []

        for name, hrs in mand_hours.items():
            idx = next((i for i,a in enumerate(ACTIVITIES_LIST) if a["name"]==name), None)
            if idx is not None:
                mand_idx.append(idx)
                mand_rem.append(hrs)

        V.cache_clear()

        total, seq = V(start_hour, energy, hunger, emotional, money, env_pref,
                       sleep_eff, -1, 0,
                       tuple(mand_idx), tuple(mand_rem),
                       -1, 0)

        if total < -1e5:
            st.error("Não foi possível gerar um planejamento viável.")
        else:
            st.success(f"Utilidade total: {total:.2f}")
            for h, n, r, c in seq:
                st.write(f"{h}:00 → {n} (U={r:.2f}, R${c:.2f})")

            fig = plot_states(seq, start_hour,
                              (start_hour, energy, hunger, emotional, money, env_pref))
            st.pyplot(fig)

# -------------------------------------------------------------------
# ABA 2 - CRUD DE ATIVIDADES
# -------------------------------------------------------------------
with tabs[1]:

    sha = get_file_sha()
    df = load_activities_df(sha)

    st.header("Cadastrar nova atividade")

    name = st.text_input("Nome")
    cost = st.number_input("Custo (R$)", 0.0, 2000.0, 0.0)
    delta_energy = st.number_input("Variação de energia (-5 a 5)", -5, 5, 0)
    delta_hunger = st.number_input("Variação de fome (-5 a 5)", -5, 5, 0)
    base_utility = st.number_input("Utilidade base (1-10)", 1, 10, 5)
    environment = st.selectbox("Ambiente", (1,2,3), format_func=lambda x: {1:"Dentro",2:"Fora",3:"Tanto faz"}[x])
    earliest_hour = st.number_input("Primeira hora possível (0-23)", 0, 23, 6)
    latest_hour = st.number_input("Última hora possível (0-23)", 0, 23, 22)
    winddown = st.radio("Adequada para o final do dia?", (True, False))

    if st.button("Cadastrar atividade"):
        if not name.strip():
            st.warning("Nome obrigatório.")
        else:
            df.loc[len(df)] = [
                name, cost, delta_energy, delta_hunger,
                base_utility, environment, earliest_hour,
                latest_hour, winddown
            ]

            ok = update_github_csv(df.to_csv(index=False), f"Adicionar {name}")

            if ok:
                st.success("Atividade cadastrada!")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("Erro ao atualizar GitHub.")

    st.markdown("---")
    st.header("Remover atividade")

    if not df.empty:
        choice = st.selectbox("Escolha", df["name"].tolist())

        if st.button("Remover"):
            new_df = df[df["name"] != choice]
            ok = update_github_csv(new_df.to_csv(index=False),
                                   f"Remover {choice}")

            if ok:
                st.success("Atividade removida!")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("Falha ao remover.")
    else:
        st.info("Nenhuma atividade cadastrada.")

# -------------------------------------------------------------------
# ABA 3 - LISTAGEM
# -------------------------------------------------------------------
with tabs[2]:

    sha = get_file_sha()
    df = load_activities_df(sha)

    st.header("Atividades cadastradas")

    if df.empty:
        st.info("Nenhuma atividade.")
    else:

        def env_label(v):
            v = int(v)
            return {1:"Dentro",2:"Fora",3:"Tanto faz"}[v]

        df_show = df.copy()
        df_show["environment"] = df_show["environment"].apply(env_label)
        df_show["winddown"] = df_show["winddown"].apply(lambda x: "✔" if str(x).lower()=="true" else "")

        st.dataframe(df_show, use_container_width=True)
