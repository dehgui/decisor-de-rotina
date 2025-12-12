# app.py - Decisor de Rotina (versão final com UI, gráfico e PDF)
import streamlit as st
import pandas as pd
import requests
import base64
from functools import lru_cache
from io import BytesIO
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------------
# CONFIGURAÇÕES
# ---------------------------
REPO_OWNER = "dehgui"
REPO_NAME = "decisor-de-rotina"
FILE_PATH = "activities.csv"
RAW_CSV_URL = "https://raw.githubusercontent.com/dehgui/decisor-de-rotina/refs/heads/main/activities.csv"

GH_TOKEN = st.secrets.get("GH_TOKEN", None)

# Model params
MAX_ENERGY = 5
MAX_HUNGER = 5
MAX_EMO = 5
DECAY_FACTOR = 0.7
WINDDOWN_PENALTY = 4.0
WINDDOWN_WINDOW = 2

# ---------------------------
# UTILIDADES GITHUB
# ---------------------------
def get_file_sha():
    """Retorna o SHA atual do arquivo no repo (necessário para PUT)."""
    if GH_TOKEN is None:
        return None
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GH_TOKEN}"}
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return r.json().get("sha")
    return None

def update_github_csv(new_content, commit_message="Atualização automática pelo app"):
    """Atualiza o arquivo CSV no GitHub (retorna True se OK)."""
    if GH_TOKEN is None:
        return False
    sha = get_file_sha()
    if sha is None:
        return False
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GH_TOKEN}"}
    encoded = base64.b64encode(new_content.encode()).decode()
    payload = {
        "message": commit_message,
        "content": encoded,
        "sha": sha
    }
    r = requests.put(url, headers=headers, json=payload)
    return r.status_code in (200, 201)

# ---------------------------
# CARREGAR ATIVIDADES (CSV do GitHub)
# ---------------------------
@st.cache_data(ttl=60)
def load_activities_df():
    try:
        df = pd.read_csv(RAW_CSV_URL)
        # garantia de colunas e tipos
        expected = ["name","cost","delta_energy","delta_hunger","base_utility",
                    "environment","earliest_hour","latest_hour","winddown"]
        for c in expected:
            if c not in df.columns:
                st.warning(f"A coluna '{c}' não existe no CSV. Verifique o arquivo no GitHub.")
                df[c] = ""  # preencher para evitar KeyError
        return df[expected]
    except Exception as e:
        st.error(f"Erro ao carregar activities.csv do GitHub: {e}")
        return pd.DataFrame(columns=[
            "name","cost","delta_energy","delta_hunger","base_utility",
            "environment","earliest_hour","latest_hour","winddown"
        ])

# ---------------------------
# FUNÇÕES DE MDP / PD
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
    except Exception:
        # fallback values
        base, cost, de, dh, env = 0.0, 0.0, 0, 0, 3

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
    new_energy = clamp(energy + int(act.get("delta_energy", 0)), 0, MAX_ENERGY)
    new_hunger = clamp(hunger + int(act.get("delta_hunger", 0)), 0, MAX_HUNGER)
    new_emotional = emotional
    new_money = max(0.0, money - float(act.get("cost", 0)))
    return (new_hour, new_energy, new_hunger, new_emotional, new_money, env_pref)

# DP with prev activity and consecutive count
@lru_cache(maxsize=None)
def V(hour, energy, hunger, emotional, money, env_pref, sleep_hour, prev_idx, consec):
    if hour >= sleep_hour:
        return (0.0, [])
    activities = ACTIVITIES_LIST  # global set before call
    best_val = -1e9
    best_seq = []
    for i, act in enumerate(activities):
        try:
            earliest = int(act.get("earliest_hour", 0))
            latest = int(act.get("latest_hour", 23))
        except Exception:
            earliest, latest = 0, 23
        if not (earliest <= hour <= latest):
            continue
        if float(act.get("cost", 0)) > money:
            continue
        base_r = base_reward(act, (hour, energy, hunger, emotional, money, env_pref))
        act_wind = str(act.get("winddown", "False")).strip().lower() == "true"
        if i == prev_idx:
            r = base_r * (DECAY_FACTOR ** consec)
        else:
            r = base_r
        if hour >= sleep_hour - WINDDOWN_WINDOW and not act_wind:
            r -= WINDDOWN_PENALTY
        nt = transition((hour, energy, hunger, emotional, money, env_pref), act)
        next_prev = i
        next_consec = consec + 1 if i == prev_idx else 1
        future_val, future_seq = V(nt[0], nt[1], nt[2], nt[3], nt[4], nt[5], sleep_hour, next_prev, next_consec)
        total = r + future_val
        if total > best_val:
            best_val = total
            best_seq = [(hour, act.get("name",""), r, float(act.get("cost",0)))] + future_seq
    # fallback: descansar
    if best_val < -1e8:
        rest_r = 1.0
        rest_act = {"name":"Descansar","delta_energy":1,"delta_hunger":0,"base_utility":1,"cost":0,"environment":1,"earliest_hour":0,"latest_hour":23,"winddown":"True"}
        nt = transition((hour, energy, hunger, emotional, money, env_pref), rest_act)
        future_val, future_seq = V(nt[0], nt[1], nt[2], nt[3], nt[4], nt[5], sleep_hour, -1, 0)
        return (rest_r + future_val, [(hour, "Descansar", rest_r, 0.0)] + future_seq)
    return (best_val, best_seq)

# ---------------------------
# FUNÇÃO PARA GERAR GRÁFICO
# ---------------------------
def plot_states(seq, start_hour, sleep_hour, initial_state):
    # seq: list of tuples (hour, name, reward, cost)
    hours = []
    energy_vals = []
    hunger_vals = []
    # simulate transitions using ACT map
    hour, energy, hunger, emotional, money, env_pref = initial_state
    hours.append(hour)
    energy_vals.append(energy)
    hunger_vals.append(hunger)
    name_map = {a['name']: a for a in ACTIVITIES_LIST}
    for (h, name, rec, cost) in seq:
        act = name_map.get(name, None)
        if act:
            energy = clamp(energy + int(act.get("delta_energy",0)), 0, MAX_ENERGY)
            hunger = clamp(hunger + int(act.get("delta_hunger",0)), 0, MAX_HUNGER)
        else:
            # fallback: assume small rest
            energy = clamp(energy + 1, 0, MAX_ENERGY)
        hours.append(h+1)
        energy_vals.append(energy)
        hunger_vals.append(hunger)
    # plot
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(hours, energy_vals, marker='o', label='Energia')
    ax.plot(hours, hunger_vals, marker='s', label='Fome')
    ax.set_xlabel("Hora")
    ax.set_xticks(range(start_hour, sleep_hour+1))
    ax.set_ylim(0, max(MAX_ENERGY, MAX_HUNGER))
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right')
    plt.tight_layout()
    return fig

# ---------------------------
# PDF / TXT EXPORT
# ---------------------------
def make_pdf_bytes(title, seq, total_util, initial_state):
    """
    Gera PDF simples com reportlab. Se reportlab não estiver disponível, gera TXT bytes.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        w, h = A4
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, h-40, title)
        c.setFont("Helvetica", 11)
        c.drawString(40, h-60, f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        c.drawString(40, h-80, f"Utilidade total esperada: {total_util:.2f}")
        y = h-110
        for hora, nome, rec, custo in seq:
            line = f"{hora}:00 -> {nome} (utilidade {rec:.2f}, custo R${custo:.2f})"
            c.drawString(40, y, line)
            y -= 16
            if y < 80:
                c.showPage()
                y = h-40
        c.save()
        buffer.seek(0)
        return buffer.read(), "application/pdf", "planejamento.pdf"
    except Exception:
        # fallback para texto
        txt = f"{title}\nGerado em: {datetime.now().isoformat()}\nUtilidade total: {total_util:.2f}\n\n"
        for hora, nome, rec, custo in seq:
            txt += f"{hora}:00 -> {nome} (utilidade {rec:.2f}, custo R${custo:.2f})\n"
        return txt.encode("utf-8"), "text/plain", "planejamento.txt"

# ---------------------------
# CSS / THEME
# ---------------------------
st.set_page_config(page_title="Decisor de Rotina", page_icon="🔮", layout="wide")
custom_css = """
<style>
/* background gradient */
.reportview-container .main {
  background: linear-gradient(180deg, #0f1724 0%, #0b1220 100%);
  color: #f7f7fb;
}
h1, h2, h3 { color: #ffd1e8; }
.stButton>button { background-color:#ff6fa3 !important; color: white !important; border-radius:8px; }
[data-testid="stSidebar"] { display: none; } /* hide default sidebar */
.stDataFrame { background: rgba(255,255,255,0.03); border-radius:8px; padding:8px; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ---------------------------
# INTERFACE PRINCIPAL (TABS)
# ---------------------------
st.title("🔮 DECISOR DE ROTINA — Planejamento Diário (MDP + PD)")

tabs = st.tabs(["📅 Gerar planejamento", "🛠 Gerenciar atividades", "📋 Atividades cadastradas"])

# ---------------------------
# ABA 1 — GERAR PLANEJAMENTO
# ---------------------------
with tabs[0]:
    st.header("Defina seu estado atual")
    # compact columns for mobile friendliness
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        start_hour = st.number_input("Hora atual", 0, 23, 9, key="start_hour")
        sleep_hour = st.number_input("Hora de dormir", 0, 23, 23, key="sleep_hour")
        env_pref = st.selectbox("Ambiente preferido", (1,2,3), format_func=lambda x: {1:"Dentro",2:"Fora",3:"Tanto faz"}[x])
    with col2:
        energy = st.slider("Energia (0-5)", 0, 5, 3, key="energy")
        hunger = st.slider("Fome (0-5)", 0, 5, 2, key="hunger")
    with col3:
        emotional = st.slider("Estado emocional (0-5)", 0, 5, 3, key="emotional")
        money = st.number_input("Dinheiro disponível (R$)", 0.0, 5000.0, 50.0, step=5.0, key="money")

    st.markdown("----")
    st.write("Toque em **Gerar planejamento** para calcular a sequência ótima de atividades até a hora de dormir.")
    df = load_activities_df()
    ACTIVITIES_LIST = df.to_dict(orient="records")

    if st.button("✨ Gerar planejamento"):
        V.cache_clear()
        val, seq = V(start_hour, energy, hunger, emotional, money, env_pref, sleep_hour, -1, 0)
        if not seq:
            st.info("Nenhuma sequência válida encontrada com as restrições atuais.")
        else:
            st.markdown(f"## Utilidade total: {val:.2f}")
            st.markdown("### Planejamento sugerido (hora → atividade):")
            for hora, nome, rec, custo in seq:
                st.markdown(f"- **{hora}:00** → {nome} (utilidade {rec:.2f}, custo R${custo:.2f})")
            # plot evolution
            fig = plot_states(seq, start_hour, sleep_hour, (start_hour, energy, hunger, emotional, money, env_pref))
            st.pyplot(fig, clear_figure=True)
            # download PDF/TXT
            content, mime, filename = make_pdf_bytes("Planejamento Diário", seq, val, (start_hour, energy, hunger, emotional, money, env_pref))
            st.download_button("📥 Baixar planejamento (PDF/TXT)", data=content, file_name=filename, mime=mime)

# ---------------------------
# ABA 2 — GERENCIAR ATIVIDADES
# ---------------------------
with tabs[1]:
    st.header("📌 Cadastrar nova atividade")
    st.write("Preencha os campos abaixo e clique em **Cadastrar atividade**. A ação atualizará o arquivo `activities.csv` no GitHub automaticamente (requer GH_TOKEN configurado).")
    name = st.text_input("Nome da atividade")
    cost = st.number_input("Custo (R$)", min_value=0.0, max_value=1000.0, value=0.0, step=1.0)
    delta_energy = st.number_input("Variação de energia (inteiro, negativo se consome)", -5, 5, 0)
    delta_hunger = st.number_input("Variação da fome (inteiro, negativo se reduz)", -5, 5, 0)
    base_utility = st.number_input("Utilidade base (1 a 10)", 1, 10, 5)
    environment = st.selectbox("Ambiente", (1,2,3), format_func=lambda x: {1:"Dentro de casa",2:"Fora de casa",3:"Tanto faz"}[x])
    earliest_hour = st.number_input("Hora inicial possível (0-23)", 0, 23, 6)
    latest_hour = st.number_input("Última hora possível (0-23)", 0, 23, 22)
    winddown = st.radio("Atividade adequada para o final do dia (winddown)?", (True, False), index=1, format_func=lambda x: "Sim" if x else "Não")
    st.caption("Explicação: 'winddown' indica que a atividade é adequada para as últimas horas antes de dormir (ex.: meditar, ler, descansar). Atividades sem winddown (estudar, correr) serão penalizadas nas últimas horas.")

    if st.button("Cadastrar atividade"):
        if GH_TOKEN is None:
            st.error("GH_TOKEN não configurado nos Secrets. Não posso atualizar o GitHub.")
        elif not name.strip():
            st.warning("Informe o nome da atividade.")
        else:
            df_new = load_activities_df()
            # append
            df_new.loc[len(df_new)] = [name, cost, delta_energy, delta_hunger, base_utility, environment, earliest_hour, latest_hour, str(winddown)]
            csv_text = df_new.to_csv(index=False)
            ok = update_github_csv(csv_text, commit_message=f"Adicionar atividade: {name}")
            if ok:
                st.success("Atividade cadastrada e CSV atualizado no GitHub ✅ (pode demorar alguns segundos para propagar).")
                st.experimental_rerun()
            else:
                st.error("Falha ao atualizar o GitHub. Verifique GH_TOKEN e permissões.")

    st.markdown("---")
    st.header("🗑 Remover atividade")
    df_remove = load_activities_df()
    if not df_remove.empty:
        choice = st.selectbox("Escolha a atividade para remover", df_remove["name"].tolist())
        if st.button("Remover atividade"):
            if GH_TOKEN is None:
                st.error("GH_TOKEN não configurado nos Secrets.")
            else:
                new_df = df_remove[df_remove["name"] != choice]
                ok = update_github_csv(new_df.to_csv(index=False), commit_message=f"Remover atividade: {choice}")
                if ok:
                    st.success("Atividade removida com sucesso do GitHub ✅.")
                    st.experimental_rerun()
                else:
                    st.error("Falha ao atualizar o GitHub. Verifique token/permissões.")
    else:
        st.info("Nenhuma atividade cadastrada para remover.")

# ---------------------------
# ABA 3 — ATIVIDADES CADASTRADAS
# ---------------------------
with tabs[2]:
    st.header("📋 Atividades cadastradas (fonte: activities.csv no GitHub)")
    df_all = load_activities_df()
    if df_all.empty:
        st.info("Nenhuma atividade cadastrada.")
    else:
        # show friendly table with emojis
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

    st.markdown("---")
    st.caption("Edite este arquivo diretamente no GitHub para mudanças manuais, ou use a aba 'Gerenciar atividades' para atualizar pelo app.")

# ---------------------------
# FIM
# ---------------------------
