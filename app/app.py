import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

    
# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FIFA World Cup 2026 simulator",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# ESTILOS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #0a0e1a; color: #e8eaf6; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #112240 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #cdd9e5 !important; }
[data-testid="stSidebar"] label { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64b5f6 !important; font-weight: 600; }

h1 { color: #ffffff !important; font-weight: 900 !important; letter-spacing: -0.03em; }
h2, h3 { color: #90caf9 !important; font-weight: 600 !important; }

.result-card {
    background: linear-gradient(135deg, #112240 0%, #0d1b2a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.result-card .label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: #64b5f6; margin-bottom: 0.25rem; }
.result-card .value { font-size: 2.2rem; font-weight: 900; color: #ffffff; line-height: 1; }
.result-card .subvalue { font-size: 0.9rem; color: #90caf9; margin-top: 0.3rem; }

.scoreboard {
    background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #1a237e 100%);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    border: 1px solid #3f51b5;
    margin-bottom: 1.5rem;
}
.scoreboard .vs { font-size: 0.9rem; color: #90caf9; text-transform: uppercase; letter-spacing: 0.2em; }
.scoreboard .teams { font-size: 1.8rem; font-weight: 900; color: #ffffff; margin: 0.4rem 0; }
.scoreboard .score { font-size: 3.5rem; font-weight: 900; color: #ffd54f; letter-spacing: 0.05em; }
.scoreboard .outcome { font-size: 1rem; color: #a5d6a7; margin-top: 0.5rem; font-weight: 600; }

.model-badge {
    display: inline-block;
    background: #1e3a5f;
    border: 1px solid #42a5f5;
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-size: 0.75rem;
    color: #90caf9;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}

hr { border-color: #1e3a5f !important; }

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #1565c0, #0d47a1);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.75rem;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 0.2s;
}
.stButton > button:hover { background: linear-gradient(135deg, #1976d2, #1565c0); transform: translateY(-1px); }

.stSelectbox > div > div { background: #112240; border-color: #1e3a5f; color: #e8eaf6; }
.stRadio > div { gap: 0.5rem; }
.stRadio label { color: #cdd9e5 !important; }
.stSlider { padding-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CARGA DE MODELOS Y EQUIPOS
# ─────────────────────────────────────────────

@st.cache_resource
def load_models():
    import pickle
    from src.ELO import EloModel, ModelLoader as EloModelLoader
    from src.DIXON_COLES import DixonColesModel, ModelLoader as DCModelLoader
    from src.loaders import load_elo, load_dc
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]

    elo_model_path = ROOT / "saved_models" / "elo_model_v1.pkl"
    dc_model_path  = ROOT / "saved_models" / "dixoncoles_model_v1.pkl"
    
    import __main__
    from src.ELO import EloModel

    __main__.EloModel = EloModel
    __main__.DixonColesModel = DixonColesModel
    
    with open(elo_model_path, 'rb') as f:
        elo_model = pickle.load(f)
    with open(dc_model_path, 'rb') as f:
        dc_model = pickle.load(f)        
    
    return {
        "Elo — Bivariate Poisson": elo_model,
        "Dixon-Coles": dc_model,
    }
    


@st.cache_data
def load_teams():
    import os
    import sys
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # 2. Ahora sí podrás usarla aquí sin errores:
    INPUT_FILE = os.path.join(SCRIPT_DIR, "..", "data", "processed", "wc_teams.csv")
    df = pd.read_csv(INPUT_FILE)
    df_ordenado = df.sort_values(by='team_name')
    return df_ordenado["team_name"].tolist()


# ─────────────────────────────────────────────
# SIMULACIÓN
# ─────────────────────────────────────────────

def run_simulation(model_name: str, team_a: str, team_b: str, n_sims: int):
    models = load_models()
    model  = models[model_name]
    rng= np.random.default_rng()
    if n_sims == 1:
        resultado = model.simulate_match(team_a, team_b,rng)
        print('aca')
    else:
        resultado = model.simular_n_partidos(team_a, team_b, n_sims)

    return {
        "prob_matrix"   : resultado["prob_matrix"],
        "resultado_esp" : resultado["resultado_esp"],
        "resultado_pred": resultado["resultado_pred"],
        "moda"          : resultado["moda"],
        "prob_moda"     : resultado["prob_moda"],
        "ganador"       : resultado["ganador"],
        "prob_local"    : resultado["prob_local"],
        "prob_empate"   : resultado["prob_empate"],
        "prob_visita"   : resultado["prob_visita"],
    }


# ─────────────────────────────────────────────
# VISUALIZACIÓN
# ─────────────────────────────────────────────

def plot_prob_matrix(prob_matrix, team_a, team_b, moda, resultado_esp):
    max_g = prob_matrix.shape[0] - 1

    fig, ax = plt.subplots(figsize=(8, 6.5))
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#0a0e1a")

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "mundial",
        ["#0a0e1a", "#0d2137", "#0d47a1", "#1976d2", "#ffd54f"],
        N=256
    )
    cmap = plt.cm.inferno
    

    pct_matrix = prob_matrix * 100
    max_prob = pct_matrix.max()

    # En lugar de usar vmin y vmax a secas, usamos una normalización que suavice el mapa
    norm = mcolors.Normalize(vmin=0, vmax=max_prob)

    im = ax.imshow(pct_matrix.T, cmap=cmap, aspect="auto",
                    norm=norm, origin="lower")

    for i in range(max_g + 1):
        for j in range(max_g + 1):
            val = pct_matrix[i, j]
            # color = "white" if val > pct_matrix.max() * 0.5 else "#90caf9"
            # weight = "bold" if (i, j) == moda else "normal"
            # ax.text(i, j, f"{val:.1f}%", ha="center", va="center",
            #         fontsize=7.5, color=color, fontweight=weight, fontfamily="monospace")
            if val > max_prob * 0.75:
                color = "#0a0e1a"  # Texto oscuro sobre fondo amarillo brillante (Moda)
            elif val > max_prob * 0.3:
                color = "white"    # Texto blanco sobre azul intermedio
            else:
                color = "#90caf9"  # Texto celeste sobre fondo muy oscuro
                
            weight = "bold" if (i, j) == moda else "normal"
            ax.text(i, j, f"{val:.1f}%", ha="center", va="center",
                    fontsize=7.5, color=color, fontweight=weight, fontfamily="monospace")

    rect = plt.Rectangle((moda[0] - 0.5, moda[1] - 0.5), 1, 1,
                          linewidth=2.5, edgecolor="#ff007f", facecolor="none", zorder=5)
    ax.add_patch(rect)

    if resultado_esp != moda:
        rect2 = plt.Rectangle((resultado_esp[0] - 0.5, resultado_esp[1] - 0.5), 1, 1,
                               linewidth=2, edgecolor="#a5d6a7", facecolor="none",
                               linestyle="--", zorder=5)
        ax.add_patch(rect2)

    ax.set_xticks(range(max_g + 1))
    ax.set_yticks(range(max_g + 1))
    ax.set_xticklabels(range(max_g + 1), color="#90caf9", fontsize=9)
    ax.set_yticklabels(range(max_g + 1), color="#90caf9", fontsize=9)
    ax.set_xlabel(f"Goals  {team_a}  (home)", color="#64b5f6", fontsize=10, labelpad=8)
    ax.set_ylabel(f"Goals  {team_b}  (away)", color="#64b5f6", fontsize=10, labelpad=8)
    ax.set_title("Score Probability Matrix", color="#ffffff", fontsize=12, fontweight="bold", pad=12)

    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a5f")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#64b5f6", labelsize=8)
    cbar.set_label("Probability (%)", color="#64b5f6", fontsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#90caf9")
    cbar.outline.set_edgecolor("#1e3a5f")

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color="#ff007f", linewidth=2, label="Mode (Most Likely Simulation Score)")]
    if resultado_esp != moda:
        legend_elements.append(
            Line2D([0], [0], color="#a5d6a7", linewidth=2, linestyle="--", label="Expected Scores")
        )
    ax.legend(handles=legend_elements, loc="upper right",
              facecolor="#112240", edgecolor="#1e3a5f", labelcolor="#cdd9e5", fontsize=8)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# APP PRINCIPAL
# ─────────────────────────────────────────────

def main():
    
    teams = load_teams()

    # ── Header ──
    st.markdown("""
        <div style="padding: 1.5rem 0 0.5rem 0;">
            <span style="font-size:0.8rem; text-transform:uppercase; letter-spacing:0.15em; color:#64b5f6; font-weight:600;">
                ⚽ FIFA WC2026
            </span>
            <h1 style="margin: 0.2rem 0 0.1rem 0; font-size: 2.2rem;">Match simulator</h1>
            <p style="color:#546e7a; font-size:0.9rem; margin:0;">
                Monte Carlo Simulation — Select teams, model, and number of runs.
            </p>
        </div>
        <hr style="margin: 1rem 0 1.5rem 0;">
    """, unsafe_allow_html=True)

    # ── Sidebar: solo recoge inputs, no ejecuta nada ──
    with st.sidebar:
        st.markdown("### ⚙️ Setup")
        st.markdown("---")

        st.markdown("**Teams**")
        default_a = teams.index("Argentina") if "Argentina" in teams else 0
        default_b = teams.index("Brasil") if "Brasil" in teams else 1
        team_a = st.selectbox("🏠 Home",     teams, index=default_a, key="team_a")
        team_b = st.selectbox("✈️ Away", teams, index=default_b, key="team_b")

        st.markdown("---")

        st.markdown("**Simulation model**")
        model_options = ["Elo — Bivariate Poisson", "Dixon-Coles"]
        model_name = st.radio("", model_options, key="model")

        st.markdown("---")

        st.markdown("**Number of Runs**")

        n_sims = st.slider("Number", min_value=100, max_value=5000,
                               value=1000, step=100, key="n_slider")
        st.caption(f"→ {n_sims:,} Monte Carlo simulations")

        st.markdown("---")
        run_btn = st.button("▶  Simulate match", key="run")

    # ══════════════════════════════════════════
    # LÓGICA PRINCIPAL — orden correcto:
    #   1. procesar el botón
    #   2. mostrar resultado o placeholder
    # ══════════════════════════════════════════

    # 1. Validación de equipos iguales (no hace return, solo avisa)
    if team_a == team_b:
        st.warning("⚠️  Choose different teams.")
        return  # único return permitido antes del botón

    # 2. Procesar el botón — SIEMPRE antes de cualquier otro return
    if run_btn:
        with st.spinner("Running..."):
            result = run_simulation(model_name, team_a, team_b, n_sims)
        st.session_state["result"] = result
        st.session_state["sim_config"] = {
            "team_a": team_a,
            "team_b": team_b,
            "model" : model_name,
            "n_sims": n_sims,
        }

    # 3. Si todavía no hay resultado, mostrar placeholder y salir
    if "result" not in st.session_state:
        st.markdown("""
            <div style="text-align:center; padding: 4rem 0; color:#546e7a;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">⚽</div>
                <p style="font-size: 1.1rem;">
                    Configure the match in the sidebar and click on
                    <strong style="color:#90caf9;">Simulate match</strong>.
                </p>
            </div>
        """, unsafe_allow_html=True)
        return

    # 4. Mostrar resultados
    result = st.session_state["result"]
    cfg    = st.session_state["sim_config"]

    ga, gb  = result["resultado_esp"]
    moda    = result["moda"]
    ganador = result["ganador"]
    print(ganador)
    ganador_texto = cfg["team_a"] if ganador == "Local" else (cfg["team_b"] if ganador == "Visitante" else "Empate")
    print(ganador_texto)
    # ── Scoreboard ──
    st.markdown(f"""
        <div class="scoreboard">
            <div class="vs">Simulation — {cfg['n_sims']:,} run{'s' if cfg['n_sims']>1 else ''}</div>
            <div class="teams">{cfg['team_a']}  ·  {cfg['team_b']}</div>
            <div class="score">{ga:.2f}  —  {gb:.2f}</div>
            <div class="outcome">Most Likely Outcome · {ganador_texto}</div>
        </div>
    """, unsafe_allow_html=True)

    # ── Heatmap + métricas ──
    col_map, col_metrics = st.columns([2, 1], gap="large")

    with col_map:
        fig = plot_prob_matrix(
            result["prob_matrix"],
            cfg["team_a"], cfg["team_b"],
            moda, result["resultado_esp"]
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_metrics:
        st.markdown("### Score Probabilities")
        st.markdown("---")

        col_l, col_e, col_v = st.columns(3)
        with col_l:
            st.markdown(f"""
                <div class="result-card">
                    <div class="label">Local</div>
                    <div class="value">{result['prob_local']*100:.1f}%</div>
                    <div class="subvalue">{cfg['team_a']}</div>
                </div>""", unsafe_allow_html=True)
        with col_e:
            st.markdown(f"""
                <div class="result-card">
                    <div class="label">Empate</div>
                    <div class="value">{result['prob_empate']*100:.1f}%</div>
                    <div class="subvalue">X</div>
                </div>""", unsafe_allow_html=True)
        with col_v:
            st.markdown(f"""
                <div class="result-card">
                    <div class="label">Visitante</div>
                    <div class="value">{result['prob_visita']*100:.1f}%</div>
                    <div class="subvalue">{cfg['team_b']}</div>
                </div>""", unsafe_allow_html=True)

        
        st.markdown("### Most Likely Score")
        st.markdown("---")

        st.markdown(f"""
            <div class="result-card">
                <div class="label">Score </div>
                <div class="value">{moda[0]} – {moda[1]}</div>
                <div class="subvalue">P = {result['prob_moda']*100:.2f}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
            <div class="result-card">
                <div class="label">Expected Simulation Outcome (E[goles])</div>
                <div class="value">{result['resultado_pred'][0]:.2f} – {result['resultado_pred'][1]:.2f}</div>
                <div class="subvalue">Simulation Average</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        model_short = cfg['model'].split("—")[0].strip()
        st.markdown(f"""
            <div style="text-align:center;">
                <div class="model-badge">{model_short}</div>
                <p style="color:#546e7a; font-size:0.75rem; margin-top:0.5rem;">
                    {cfg['n_sims']:,} simulation{'s' if cfg['n_sims']>1 else ''}
                </p>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Top 5 scores")

        pm   = result["prob_matrix"]
        rows = [{"Score": f"{i} – {j}", "Prob (%)": round(pm[i, j] * 100, 2)}
                for i in range(pm.shape[0]) for j in range(pm.shape[1])]
        df_top = pd.DataFrame(rows).nlargest(5, "Prob (%)").reset_index(drop=True)
        df_top.index += 1
        st.dataframe(
            df_top,
            use_container_width=True,
            hide_index=False,
            column_config={
                "Prob (%)": st.column_config.ProgressColumn(
                    "Prob (%)", format="%.2f%%",
                    min_value=0, max_value=df_top["Prob (%)"].max()
                )
            }
        )

    st.markdown("---")
    st.markdown(f"""
        <p style="color:#37474f; font-size:0.75rem; text-align:center;">
            Monte Carlo simulation· {cfg['model'].split('—')[0].strip()} · {cfg['n_sims']:,} runs
        </p>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
