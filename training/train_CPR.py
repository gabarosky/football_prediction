# -*- coding: utf-8 -*-
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.optimize import minimize
from src.CPR import CPRModel
import os
import sys
import networkx as nx
# --- LOGGING SETUP ---

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOG_DIR = Path("../logs")
LOG_DIR.mkdir(exist_ok=True)
log_name = f"CPR_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(LOG_DIR / log_name, delay=True), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# =========================================================================
# 1. FUNCIÓN DE CONSTRUCCIÓN DE GRAFO BASADO EN GOLES Y TIEMPO
# =========================================================================
def generar_dataset_con_goles_y_lambda(df_base, lmbda):
    fecha_referencia = df_base["date"].max()
    df_temp = df_base.copy()

    # Calcular el peso temporal decreciente (Memoria del modelo)
    df_temp["dias"] = (fecha_referencia - df_temp["date"]).dt.days
    df_temp["peso_temporal"] = np.exp(-lmbda * df_temp["dias"])

    # Separar los partidos para estructurar el flujo de goles
    df_l = df_temp[df_temp["home_score"] > df_temp["away_score"]].copy()
    df_v = df_temp[df_temp["away_score"] > df_temp["home_score"]].copy()
    df_e = df_temp[df_temp["home_score"] == df_temp["away_score"]].copy()

    # --- Victorias Locales ---
    # El flujo de goles va de Perdedor -> Ganador. 
    # El peso es (Goles del Ganador) * peso_temporal
    df_l["origen"] = df_l["away_team"]
    df_l["destino"] = df_l["home_team"]
    df_l["w_f"] = df_l["home_score"] * df_l["peso_temporal"]

    # --- Victorias Visitantes ---
    df_v["origen"] = df_v["home_team"]
    df_v["destino"] = df_v["away_team"]
    df_v["w_f"] = df_v["away_score"] * df_v["peso_temporal"]

    # --- Empates ---
    # En caso de empate (ej. 2-2), ambos se convirtieron goles.
    # El flujo va en ambos sentidos ponderado por los goles anotados.
    # Si fue 0-0, le asignamos un valor mínimo simbólico (ej. 0.5) para que el arco exista.
    e1 = df_e.copy()
    e1["origen"] = e1["away_team"]
    e1["destino"] = e1["home_team"]
    e1["w_f"] = np.where(e1["home_score"] > 0, e1["home_score"], 0.5) * e1["peso_temporal"]

    e2 = df_e.copy()
    e2["origen"] = e2["home_team"]
    e2["destino"] = e2["away_team"]
    e2["w_f"] = np.where(e2["away_score"] > 0, e2["away_score"], 0.5) * e2["peso_temporal"]

    # Consolidar y agrupar todos los arcos de goles
    df_todos_arcos = pd.concat([df_l, df_v, e1, e2], ignore_index=True)
    df_g = df_todos_arcos.groupby(["origen", "destino"])["w_f"].sum().reset_index()

    # Construir Grafo Dirigido con NetworkX
    G = nx.from_pandas_edgelist(
        df_g, source="origen", target="destino", edge_attr="w_f", create_using=nx.DiGraph()
    )
    pr_dict = nx.pagerank(G, weight="w_f")

    # Mapear PageRank a escala ELO (* 10,000)
    media_pr = np.mean(list(pr_dict.values()))
    elo_map = {k: (v * 10000) for k, v in pr_dict.items()}

    # Asignar los ELOS basados en goles al DataFrame de trabajo
    df_temp["elo_home"] = df_temp["home_team"].map(elo_map).fillna(media_pr * 10000)
    df_temp["elo_away"] = df_temp["away_team"].map(elo_map).fillna(media_pr * 10000)

    return df_temp, elo_map


# =========================================================================
# 2. FUNCIÓN DE PÉRDIDA DIRECTA DE RPS (Métrica Objetivo)
# =========================================================================
def loss_directa_rps(params, df_partidos):
    divisor, c, w = params

    # Restricciones duras para evitar valores indefinidos
    if divisor <= 10 or w <= 0:
        return np.inf

    elo_diff = df_partidos["elo_home"].values - df_partidos["elo_away"].values

    # Tus ecuaciones sigmoidales con la ventana de empate 'w'
    p_local = 1 / (1 + 10 ** (-(elo_diff + c - w) / divisor))
    p_visita = 1 / (1 + 10 ** ((elo_diff + c + w) / divisor))
    p_empate = 1 - p_local - p_visita

    # Penalización si las probabilidades colapsan o se vuelven negativas
    if np.any(p_empate <= 0):
        return np.inf

    # Sumas acumuladas (CDF) para el cálculo de RPS
    p_cum1 = p_local
    p_cum2 = p_local + p_empate

    r_cum1 = df_partidos["gano_local"].values
    r_cum2 = df_partidos["gano_local"].values + df_partidos["fue_empate"].values

    # Fórmula del Ranked Probability Score (RPS) para 3 categorías
    rps_partidos = ((p_cum1 - r_cum1) ** 2 + (p_cum2 - r_cum2) ** 2) / 2

    return np.mean(rps_partidos)


def run_pipeline():
    # 1. Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Correctly goes up one level, then into data/processed/
    INPUT_FILE = os.path.join(SCRIPT_DIR, "..", "data", "processed", "results.parquet")
    MODEL_OUTPUT = Path(os.path.join(SCRIPT_DIR, "..", "saved_models", "CPR_model_v1.pkl"))
    MODEL_OUTPUT.parent.mkdir(exist_ok=True)

    # 2. Load and Prepare
    logger.info("Loading processed data...")
    df = pd.read_parquet(INPUT_FILE)
    all_teams = sorted(list(set(df['home_team']).union(set(df['away_team']))))
    team_to_id = {team: i for i, team in enumerate(all_teams)}
    
    df['h_id'] = df['home_team'].map(team_to_id)
    df['a_id'] = df['away_team'].map(team_to_id)
    
    TRAIN_END = 21715
    data_array = df.iloc[:TRAIN_END][['h_id', 'a_id', 'home_score', 'away_score']].values.astype(int)
    
    # 3. Optimize
    df["gano_local"] = (df["home_score"] > df["away_score"]).astype(int)
    df["fue_empate"] = (df["home_score"] == df["away_score"]).astype(int)
    df["gano_visita"] = (df["home_score"] < df["away_score"]).astype(int)
    
    # Definimos la grilla de búsqueda para encontrar el lambda óptimo
    lambdas_grid = [0.0001, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005]
    lambdas_grid = np.linspace(0.000001, 0.0001, 10)
    
    mejor_rps_global = np.inf
    mejores_params_internos = None
    mejor_lambda = None
    mejor_CPR_dict = None
    
    print("Iniciando optimización basada en GOLES y minimización de RPS...")
    print("-" * 65)
    
    for lmbda_candidato in lambdas_grid:
        # A. Construir el grafo de goles y obtener los ELOS para este lambda
        df_con_goles, CPR_dict = generar_dataset_con_goles_y_lambda(df, lmbda_candidato)
    
        # B. Optimizar las perillas de la ecuación de predicción (divisor, c, w)
        initial_guess = [400.0, 50.0, 80.0]
        bounds = [(100, 2000), (-500, 500), (5, 500)]
    
        resultado = minimize(
            loss_directa_rps,
            initial_guess,
            args=(df_con_goles,),
            method="L-BFGS-B",
            bounds=bounds,
        )
    
        rps_actual = resultado.fun
        print(f"Lambda: {lmbda_candidato:<7} | Mínimo RPS alcanzado: {rps_actual:.5f}")
    
        # C. Registrar el mejor absoluto
        if rps_actual < mejor_rps_global:
            mejor_rps_global = rps_actual
            mejor_lambda = lmbda_candidato
            mejores_params_internos = resultado.x  # Guarda [divisor, c, w]
            mejor_CPR_dict = CPR_dict
    
    # Desempaquetar los parámetros ganadores del pipeline
    div_opt, c_opt, w_opt = mejores_params_internos
    
    print("=" * 65)
    print("         RESULTADOS DE LA OPTIMIZACIÓN BASADA EN GOLES        ")
    print("=============================================================")
    print(f"1. Lambda Óptimo (Decaimiento temporal):  {mejor_lambda}")
    print(f"2. Divisor Óptimo (Escala de las curvas): {div_opt:.2f}")
    print(f"3. Ventaja de Localía Óptima (c):         {c_opt:.2f}")
    print(f"4. Ventana de Empate Óptima (w):          {w_opt:.2f}")
    print(f"--> MENOR RPS PROMEDIO ALCANZADO:         {mejor_rps_global:.5f}")
    print("=============================================================")
    
    logger.info(f"Best Params: div={div_opt:.2f}, draw={w_opt:.2f}, Margin={c_opt:.2f}")

    
    # 5. Serialize Model
    # Note: We set HFA to 0 for neutral ground predictions in the final model state
    optimized_params = [div_opt,c_opt, w_opt] 
    
    
    id_to_team = {v: k for k, v in team_to_id.items()}

# 2. Generamos la lista de CPRs ordenados de menor a mayor ID (0, 1, 2...)
# Usamos .get() por seguridad en caso de que algún ID falte en los datos
    lista_cpr = [
        mejor_CPR_dict[id_to_team[i]] 
        for i in range(len(team_to_id))
        ]
    
    model_obj = CPRModel(lista_cpr, team_to_id, optimized_params)
    
    print(model_obj.__class__.__module__)
    with open(MODEL_OUTPUT, 'wb') as f:
        pickle.dump(model_obj, f)
    
    logger.info(f"Model object saved successfully to {MODEL_OUTPUT}")
    
if __name__ == "__main__":
    try:
        run_pipeline()
    finally:
        logging.shutdown()
            