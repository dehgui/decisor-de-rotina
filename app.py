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
RAW_CSV_URL = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/refs/heads/main/{FILE_PATH}"
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

def update_github_csv(new_content, commit_message="Atualização automática"):
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
    base = float(act.get("base_utility", 0))
    cost = float(act.get("cost", 0))
    de = int(act.get("delta_energy", 0))
    dh = int(act.get("delta_hunger", 0))
    env = int(act.get("environment", 3))
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
        e0 = int(act["earliest_hour"])
        e1 = int(act["latest_hour"])
        if not (e0 <= hour_mod <= e1):
            return (-1e6, [])
        if float(act["cost"]) > money:
            return (-1e6, [])
        base_r = base_reward(act, (hour, energy, hunger, emotional, money, env_pref))
        r = base_r * (DECAY_FACTOR ** consec) if i == prev_idx else base_r
        act_wind = str(act["winddown"]).lower() == "true"
        if hour >= sleep_hour - WINDDOWN_WINDOW and not act_wind:
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

    for i, act in enumerate(activities):
        hour_mod = hour % 24
        e0 = int(act["earliest_hour"])
        e1 = int(act["latest_hour"])
        if not (e0 <= hour_mod <= e1):
            continue
        if float(act["cost"]) > money:
            continue

        is_mand = False
        mand_pos = -1
        for p, idx in enumerate(mand_idx_tuple):
            if idx == i:
                is_mand = True
                mand_pos = p
                break

        if total_mand > remaining:
            continue
        if total_mand == remaining and not is_mand:
            continue

        base_r = base_reward(act, (hour, energy, hunger, emotional, money, env_pref))
        r = base_r * (DECAY_FACTOR ** consec) if i == prev_idx else base_r
        act_wind = str(act["winddown"]).lower() == "true"
        if hour >= sleep_hour - WINDDOWN_WINDOW and not act_wind:
            r -= WINDDOWN_PENALTY

        mand_rem = list(mand_rem_tuple)
        next_pos = -1
        next_consec = 0

        if is_mand:
            mand_rem[mand_pos] -= 1
            if mand_rem[mand_pos] > 0:
                next_pos = mand_pos
                next_consec = 1

        nt = transition((hour, energy, hunger, emotional, money, env_pref), act)
        fv, fs = V(nt[0], nt[1], nt[2], nt[3], nt[4], nt[5],
                    sleep_hour, i, (consec+1 if i==prev_idx else 1),
                    mand_idx_tuple, tuple(mand_rem),
                    next_pos, next_consec)

        s = r + fv
        if s > best_val:
            best_val = s
            best_seq = [(hour, act["name"], r, float(act["cost"]))] + fs

    return (best_val, best_seq)

def plot_states(seq, start_hour, sleep_hour, initial_state):
    hours = []
    ev = []
    hv = []
    hour, energy, hunger, emo, money, env_pref = initial_state
    hours.append(hour)
    ev.append(energy)
    hv.append(hunger)
    lookup = {a["name"]: a for a in ACTIVITIES_LIST}
    for h, name, r, c in seq:
        act = lookup.get(name)
        if act:
            energy = clamp(energy + int(act["delta_energy"]), 0, MAX_ENERGY)
            hunger = clamp(hunger + int(act["delta_hunger"]), 0, MAX_HUNGER)
        hours.append(h+1)
        ev.append(energy)
        hv.append(hunger)
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(hours, ev, marker='o', label="Energia")
    ax.plot(hours, hv, marker='s', label="Fome")
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
            c.drawString(40, y, f"{hora}:00 → {nome} (U={rec:.2f}, R${custo:.2f})")
            y -= 16
            if y < 80:
                c.showPage()
                y = h-40
        c.save()
        buf.seek(0)
        return buf.read(), "application/pdf", "planejamento.pdf"
    except:
        txt = f"Utilidade total: {total_util:.2f}\n\n"
        for hora, nome, rec, custo in seq:
            txt += f"{hora}:00 → {nome} (U {rec:.2f}, R${custo:.2f})\n"
        return txt.encode(), "text/plain", "planejamento.txt"

st.set_page_config(page_title="Decisor de Rotina", layout="wide", page_icon="🔮")

st.title("🔮 DECISOR DE ROTINA — Planejamento Diário")

tabs = st.tabs(["📅 Gerar planejamento", "🛠 Gerenciar atividades", "📋 Atividades cadastradas"])

with tabs[0]:
    st.header("Defina seu estado atual")
    col1, col2, col3 = st.columns(3)
    with col1:
        start_hour = st.number_input("Hora atual (0-23)", 0, 23, 9)
        sleep_hour = st.number_input("Hora de dormir (0-23)", 0, 23, 23)
        sleep_eff = sleep_hour + 24 if sleep_hour <= start_hour else sleep_hour
        env_pref = st.selectbox("Ambiente preferido", (1,2,3),
                                format_func=lambda x: {1:"Dentro",2:"Fora",3:"Tanto faz"}[x])
    with col2:
