import streamlit as st
import pandas as pd
import requests
import base64
from functools import lru_cache
from io import BytesIO
import matplotlib.pyplot as plt

REPO_OWNER = "dehgui"
REPO_NAME = "decisor-de-rotina"
FILE_PATH = "activities.csv"
RAW_CSV_URL = f"https://raw.githubusercontent.com/dehgui/decisor-de-rotina/refs/heads/main/activities.csv"
GH_TOKEN = st.secrets.get("GH_TOKEN", None)

def get_file_sha():
    if GH_TOKEN is None:
        return None
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GH_TOKEN}"}
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return r.json().get("sha")
    return None

def update_github_csv(new_content, commit_message="Atualizacao automatica"):
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

@st.cache_data
def load_activities_df():
    try:
        df = pd.read_csv(RAW_CSV_URL)
    except:
        df = pd.DataFrame(columns=[
            "name","cost","delta_energy","delta_hunger",
            "base_utility","environment","earliest_hour",
            "latest_hour","winddown"
        ])
    expected = [
        "name","cost","delta_energy","delta_hunger",
        "base_utility","environment","earliest_hour",
        "latest_hour","winddown"
    ]
    for c in expected:
        if c not in df.columns:
            df[c] = ""
    return df[expected]

MAX_ENERGY = 5
MAX_HUNGER = 5
DECAY_FACTOR = 0.7
WINDDOWN_PENALTY = 4.0
WINDDOWN_WINDOW = 2

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

from functools import lru_cache
@lru_cache(maxsize=None)
def V(hour, energy, hunger, emotional, money, env_pref,
      sleep_hour, prev_idx, consec,
      mand_idx_tuple, mand_rem_tuple,
      current_mand_pos, current_mand_consec):

    if hour >= sleep_hour:
        if any(x > 0 for x in mand_rem_tuple):
            return (-1e6, [])
        return (0.0, [])

    activities = ACTIVITIES_LIST
    remaining = sleep_hour - hour
    total_mand = sum(mand_rem_tuple)

    best_val = -1e9
    best_seq = []

    if current_mand_pos != -1:
        i = mand_idx_tuple[current_mand_pos]
        act = activities[i]
        hour_mod = hour % 24
        try:
            e0 = int(act.get("earliest_hour", 0))
            e1 = int(act.get("latest_hour", 23))
        except:
            e0, e1 = 0, 23
        if not (e0 <= hour_mod <= e1):
            return (-1e6, [])
        if float(act.get("cost", 0)) > money:
            return (-1e6, [])
        base_r = base_reward(act, (hour, energy, hunger, emotional, money, env_pref))
        r = base_r * (DECAY_FACTOR ** consec) if i == prev_idx else base_r
        act_wind = str(act.get("winddown","")).strip().lower() == "true"
        if hour >= sleep_hour - WINDDOWN_WINDOW and not act_wind:
            r -= WINDDOWN_PENALTY
        mand_rem = list(mand_rem_tuple)
        mand_rem[current_mand_pos] = max(0, mand_rem[current_mand_pos] - 1)
        next_pos = current_mand_pos if mand_rem[current_mand_pos] > 0 else -1
        next_consec = current_mand_consec + 1 if next_pos == current_mand_pos else 0
        nt = transition((hour, energy, hunger, emotional, money, env_pref), act)
        future_val, future_seq = V(nt[0], nt[1], nt[2], nt[3], nt[4], nt[5], sleep_hour,
                                   i, next_consec,
                                   mand_idx_tuple, tuple(mand_rem), next_pos, next_consec)
        total = r + future_val
        return (total, [(hour, act.get("name",""), r, float(act.get("cost",0)))] + future_seq)

    for i, act in enumerate(activities):
        hour_mod = hour % 24
        try:
            earliest = int(act.get("earliest_hour", 0))
            latest = int(act.get("latest_hour", 23))
        except:
            earliest, latest = 0, 23
        if not (earliest <= hour_mod <= latest):
            continue
        if float(act.get("cost", 0)) > money:
            continue
        is_mand = False
        mand_pos_for_this = -1
        if mand_idx_tuple:
            for pos, mand_act_index in enumerate(mand_idx_tuple):
                if mand_act_index == i:
                    is_mand = True
                    mand_pos_for_this = pos
                    break
        if total_mand > remaining:
            continue
        if total_mand == remaining and not is_mand:
            continue
        base_r = base_reward(act, (hour, energy, hunger, emotional, money, env_pref))
        if i == prev_idx:
            r = base_r * (DECAY_FACTOR ** consec)
        else:
            r = base_r
        act_wind = str(act.get("winddown","False")).strip().lower() == "true"
        if hour >= sleep_hour - WINDDOWN_WINDOW and not act_wind:
            r -= WINDDOWN_PENALTY
        mand_rem = list(mand_rem_tuple)
        next_current_pos = -1
        next_current_consec = 0
        if is_mand:
            mand_rem[mand_pos_for_this] = max(0, mand_rem[mand_pos_for_this] - 1)
            if mand_rem[mand_pos_for_this] > 0:
                next_current_pos = mand_pos_for_this
                next_current_consec = 1
            else:
                next_current_pos = -1
                next_current_consec = 0
        nt = transition((hour, energy, hunger, emotional, money, env_pref), act)
        future_val, future_seq = V(nt[0], nt[1], nt[2], nt[3], nt[4], nt[5], sleep_hour,
                                   i, (consec+1 if i==prev_idx else 1),
                                   mand_idx_tuple, tuple(mand_rem), next_current_pos, next_current_consec)
        total = r + future_val
        if total > best_val:
            best_val = total
            best_seq = [(hour, act.get("name",""), r, float(act.get("cost",0)))] + future_seq

    return (best_val, best_seq)

def plot_states(seq, start_hour, initial_state):
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
    ax.set_xlabel("Hora")
    ax.legend()
    plt.tight_layout()
    return fig

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

st.set_page_config(page_title="Decisor de Rotina", layout="wide", page_icon="🔮")
st.title("Decisor de Rotina - Planejamento Diario")
tabs = st.tabs(["Gerar planejamento","Gerenciar atividades","Atividades cadastradas"])

with tabs[0]:
    st.header("Defina seu estado atual")
    col1, col2, col3 = st.columns(3)
    with col1:
        start_hour = st.number_input("Hora atual (0-23)", 0, 23, 9)
        sleep_hour = st.number_input("Hora de dormir (0-23)", 0, 23, 23)
        sleep_hour_effective = sleep_hour + 24 if sleep_hour <= start_hour else sleep_hour
        env_pref = st.selectbox("Ambiente preferido", (1,2,3), format_func=lambda x: {1:"Dentro",2:"Fora",3:"Tanto faz"}[x])
    with col2:
        energy = st.slider("Energia (0-5)", 0, 5, 3)
        hunger = st.slider("Fome (0-5)", 0, 5, 2)
    with col3:
        emotional = st.slider("Estado emocional (0-5)", 0, 5, 3)
        money = st.number_input("Dinheiro disponivel (R$)", 0.0, 2000.0, 50.0)

    st.markdown("---")
    df_act = load_activities_df()
    ACTIVITIES_LIST = df_act.to_dict(orient="records")
    st.subheader("Atividades obrigatorias (contiguas)")
    names = df_act["name"].tolist()
    mand_selected = st.multiselect("Selecione atividades obrigatorias", names)
    mand_hours = {}
    for n in mand_selected:
        mand_hours[n] = st.number_input(f"Horas continuas minimas para {n}", min_value=1, max_value=24, value=1, key=f"mand_{n}")

    if st.button("Gerar planejamento otimo"):
        mand_idx = []
        mand_rem = []
        for name, hours in mand_hours.items():
            idx = next((i for i,a in enumerate(ACTIVITIES_LIST) if a["name"]==name), None)
            if idx is not None:
                mand_idx.append(idx)
                mand_rem.append(hours)
        mand_idx_tuple = tuple(mand_idx)
        mand_rem_tuple = tuple(mand_rem)
        V.cache_clear()
        total, seq = V(start_hour, energy, hunger, emotional, money, env_pref,
                       sleep_hour_effective, -1, 0,
                       mand_idx_tuple, mand_rem_tuple, -1, 0)
        if total < -1e5 or not seq:
            st.error("Nao foi possivel gerar um planejamento viavel.")
        else:
            st.subheader(f"Utilidade total: {total:.2f}")
            for h, n, r, c in seq:
                st.write(f"- {h}:00 -> {n} (U {r:.2f}, R${c:.2f})")
            fig = plot_states(seq, start_hour, (start_hour, energy, hunger, emotional, money, env_pref))
            st.pyplot(fig)
            content, mime, fname = make_pdf_bytes("Planejamento Diario", seq, total)
            st.download_button("Baixar planejamento", content, file_name=fname, mime=mime)

with tabs[1]:
    st.header("Cadastrar nova atividade")
    name = st.text_input("Nome")
    cost = st.number_input("Custo (R$)", 0.0, 2000.0, 0.0)
    delta_energy = st.number_input("Variacao de energia (-5 a 5)", -5, 5, 0)
    delta_hunger = st.number_input("Variacao de fome (-5 a 5)", -5, 5, 0)
    base_utility = st.number_input("Utilidade base (1-10)", 1, 10, 5)
    environment = st.selectbox("Ambiente", (1,2,3), format_func=lambda x: {1:"Dentro",2:"Fora",3:"Tanto faz"}[x])
    earliest_hour = st.number_input("Primeira hora possivel (0-23)", 0, 23, 6)
    latest_hour = st.number_input("Ultima hora possivel (0-23)", 0, 23, 22)
    winddown = st.radio("Adequada para final do dia?", (True, False), index=1)

    if st.button("Cadastrar atividade"):
        if not name.strip():
            st.warning("Nome e obrigatorio.")
        else:
            df = load_activities_df()
            df.loc[len(df)] = [name, cost, delta_energy, delta_hunger, base_utility, environment, earliest_hour, latest_hour, str(winddown)]
            ok = update_github_csv(df.to_csv(index=False), commit_message=f"Adicionar atividade: {name}")
            if ok:
                st.success("Atividade cadastrada!")
                load_activities_df.clear()
                st.rerun()
            else:
                st.error("Falha ao atualizar GitHub.")

    st.markdown("---")
    st.header("Remover atividade")
    df_rem = load_activities_df()
    if not df_rem.empty:
        choice = st.selectbox("Escolha a atividade", df_rem["name"].tolist())
        if st.button("Remover"):
            new_df = df_rem[df_rem["name"] != choice]
            ok = update_github_csv(new_df.to_csv(index=False), commit_message=f"Remover atividade: {choice}")
            if ok:
                st.success("Atividade removida!")
                load_activities_df.clear()
                st.rerun()
            else:
                st.error("Falha ao remover.")
    else:
        st.info("Nenhuma atividade cadastrada.")

with tabs[2]:
    st.header("Atividades cadastradas")
    df_all = load_activities_df()
    if df_all.empty:
        st.info("Nenhuma atividade.")
    else:
        def env_label(v):
            try:
                v = int(v)
            except:
                v = 3
            return {1:"Dentro", 2:"Fora", 3:"Tanto faz"}[v]
        df_show = df_all.copy()
        df_show["environment"] = df_show["environment"].apply(env_label)
        df_show["winddown"] = df_show["winddown"].apply(lambda x: "✅" if str(x).lower()=="true" else " ")
        st.dataframe(df_show, use_container_width=True)
