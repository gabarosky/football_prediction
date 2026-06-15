# -*- coding: utf-8 -*-
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.optimize import minimize
from scipy.special import gammaln
import math
from src.DPOP import DPOPModel
import os
import sys
import networkx as nx
# --- LOGGING SETUP ---

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOG_DIR = Path("../logs")
LOG_DIR.mkdir(exist_ok=True)
log_name = f"DPOP_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(LOG_DIR / log_name, delay=True), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)



# =========================================================================
# 1. MOTOR ESTRUCTURAL DE POTENCIAS (PUNTO FIJO)
# =========================================================================
def calcular_potencias_estaticas(df_train, lmbda, unique_teams, max_iter=100, tol=1e-5):
    """Calcula las potencias del grafo fijando una foto estática del train set."""
    fecha_referencia = df_train["date"].max()
    df_temp = df_train.copy()

    df_temp["dias"] = (fecha_referencia - df_temp["date"]).dt.days
    df_temp["peso_temporal"] = np.exp(-lmbda * df_temp["dias"])

    df_temp["home_goles_w"] = df_temp["home_score"] * df_temp["peso_temporal"]
    df_temp["away_goles_w"] = df_temp["away_score"] * df_temp["peso_temporal"]

    df_g_home = df_temp.groupby(["home_team", "away_team"]).agg(
        goles_w=("home_goles_w", "sum"), partidos_w=("peso_temporal", "sum")
    ).reset_index()
    df_g_home.columns = ["ataca", "defiende", "goles_w", "partidos_w"]

    df_g_away = df_temp.groupby(["away_team", "home_team"]).agg(
        goles_w=("away_goles_w", "sum"), partidos_w=("peso_temporal", "sum")
    ).reset_index()
    df_g_away.columns = ["ataca", "defiende", "goles_w", "partidos_w"]

    df_goles = pd.concat([df_g_home, df_g_away], ignore_index=True)
    df_goles["goles_promedio_w"] = df_goles["goles_w"] / df_goles["partidos_w"]

    pg_n = pd.Series(1.0, index=unique_teams)
    pd_n = pd.Series(1.0, index=unique_teams)

    for _ in range(max_iter):
        pg_old = pg_n.copy()
        pd_old = pd_n.copy()

        df_goles["pd_rival_n"] = df_goles["defiende"].map(pd_old)
        df_goles["contrib_ataque"] = df_goles["goles_promedio_w"] * df_goles["pd_rival_n"]
        pg = df_goles.groupby("ataca")["contrib_ataque"].sum().reindex(unique_teams, fill_value=0.0)
        max_pg = pg.max()
        pg_n = pg / max_pg if max_pg > 0 else pg

        df_goles["pg_rival_n"] = df_goles["ataca"].map(pg_n)
        df_goles["pg_rival_n"] = df_goles["pg_rival_n"].clip(lower=0.01)
        df_goles["contrib_daño"] = df_goles["goles_promedio_w"] / df_goles["pg_rival_n"]
        daño_recibido = df_goles.groupby("defiende")["contrib_daño"].sum().reindex(unique_teams, fill_value=0.0)

        pd_vec = 1.0 / (1.0 + daño_recibido)
        max_pd = pd_vec.max()
        pd_n = pd_vec / max_pd if max_pd > 0 else pd_vec

        if np.max(np.abs(pg_old - pg_n)) < tol and np.max(np.abs(pd_old - pd_n)) < tol:
            break

    return pg_n.to_dict(), pd_n.to_dict()


# =========================================================================
# 2. LOG-LIKELIHOOD DE DIXON-COLES (MÁXIMA VEROSIMILITUD)
# =========================================================================
def log_likelihood_global_dixon_coles(params, df_train):
    k1, k2, gamma, rho = params

    pg_h = df_train["pg_home"].clip(lower=0.01).values
    pd_h = df_train["pd_home"].clip(lower=0.01).values
    pg_a = df_train["pg_away"].clip(lower=0.01).values
    pd_a = df_train["pd_away"].clip(lower=0.01).values

    x = df_train["home_score"].values
    y = df_train["away_score"].values

    lambdas = np.clip(np.exp(gamma) * (pg_h**k1) / (pd_a**k2), 1e-3, 15.0)
    mus = np.clip((pg_a**k1) / (pd_h**k2), 1e-3, 15.0)

    taus = np.ones(len(df_train))
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    taus[m00] = 1.0 - lambdas[m00] * mus[m00] * rho
    taus[m01] = 1.0 + lambdas[m01] * rho
    taus[m10] = 1.0 + mus[m10] * rho
    taus[m11] = 1.0 - rho

    violaciones = np.minimum(taus - 1e-4, 0.0)
    penalizacion = np.sum(violaciones**2) * 10000.0
    taus = np.clip(taus, 1e-6, None)

    log_p_home = x * np.log(lambdas) - lambdas - gammaln(x + 1)
    log_p_away = y * np.log(mus) - mus - gammaln(y + 1)
    log_tau = np.log(taus)

    return -np.sum(log_p_home + log_p_away + log_tau) + penalizacion


# =========================================================================
# 3. CALCULADOR DE PROBABILIDADES EXACTAS TRINARIAS
# =========================================================================
def predecir_probabilidades_exactas(lx, my, rho, max_goals=12):
    p_matrix = np.zeros((max_goals, max_goals))
    for x in range(max_goals):
        for y in range(max_goals):
            p_x = (lx**x) * np.exp(-lx) / math.factorial(x)
            p_y = (my**y) * np.exp(-my) / math.factorial(y)

            tau = 1.0
            if x == 0 and y == 0: tau = 1.0 - lx * my * rho
            elif x == 0 and y == 1: tau = 1.0 + lx * rho
            elif x == 1 and y == 0: tau = 1.0 + my * rho
            elif x == 1 and y == 1: tau = 1.0 - rho

            p_matrix[x, y] = max(0.0, p_x * p_y * tau)

    p_matrix /= p_matrix.sum()
    p_home = np.sum(np.tril(p_matrix, -1))
    p_draw = np.sum(np.diagonal(p_matrix))
    p_away = np.sum(np.triu(p_matrix, 1))

    idx_max = np.unravel_index(np.argmax(p_matrix), p_matrix.shape)
    return p_home, p_draw, p_away, idx_max[0], idx_max[1], p_matrix[idx_max]


def run_pipeline():
    # 1. Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Correctly goes up one level, then into data/processed/
    INPUT_FILE = os.path.join(SCRIPT_DIR, "..", "data", "processed", "results.parquet")
    MODEL_OUTPUT = Path(os.path.join(SCRIPT_DIR, "..", "saved_models", "DPOP_model_v2.pkl"))
    MODEL_OUTPUT.parent.mkdir(exist_ok=True)

    # 2. Load and Prepare
    logger.info("Loading processed data...")
    df = pd.read_parquet(INPUT_FILE)
    all_teams = sorted(list(set(df['home_team']).union(set(df['away_team']))))
    team_to_id = {team: i for i, team in enumerate(all_teams)}
    unique_teams = sorted(list(set(df['home_team']).union(set(df['away_team']))))
    df['date'] = pd.to_datetime(df['date'])

    # --- FASE 1: AISLAR EL PASADO HISTÓRICO (TRAIN SET) ---
    WC_2022_BEG = 24165
    df_train_base = df.iloc[:WC_2022_BEG].copy()
    
    mejor_ll = np.inf
    mejor_lambda = None
    mejores_params = None
    mejores_dict_pg, mejores_dict_pd = None, None

    print(f"--- FASE 1: Calibrando en Train Set ({len(df_train_base)} partidos) ---")
    
    lambdas_grid = np.linspace(0.000000000000001, 0.01, 10)
    for lmbda in lambdas_grid:
        # A. Resolver Potencias de Grafo para este Lambda sobre Train
        dict_pg, dict_pd = calcular_potencias_estaticas(df_train_base, lmbda, unique_teams)
        
        df_train_eval = df_train_base.copy()
        df_train_eval["pg_home"] = df_train_eval["home_team"].map(dict_pg)
        df_train_eval["pd_home"] = df_train_eval["home_team"].map(dict_pd)
        df_train_eval["pg_away"] = df_train_eval["away_team"].map(dict_pg)
        df_train_eval["pd_away"] = df_train_eval["away_team"].map(dict_pd)

        # B. Optimizar por Máxima Verosimilitud k1, k2, gamma, rho
        initial_guess = [1.0, 1.0, 0.2, 0.0]
        bounds = [(0.05, 3.5), (0.05, 3.5), (-1.0, 1.0), (-0.08, 0.12)]
        
        res = minimize(
            fun=log_likelihood_global_dixon_coles,
            x0=initial_guess,
            args=(df_train_eval,),
            method='L-BFGS-B',
            bounds=bounds
        )
        
        if res.success:
            ll_actual = res.fun
            print(f"Lambda: {lmbda:<7.5f} | NegLogLikelihood: {ll_actual:.2f} | k1={res.x[0]:.2f}, k2={res.x[1]:.2f}, γ={res.x[2]:.2f}, ρ={res.x[3]:.3f}")
            
            if ll_actual < mejor_ll:
                mejor_ll = ll_actual
                mejor_lambda = lmbda
                mejores_params = res.x
                mejores_dict_pg = dict_pg
                mejores_dict_pd = dict_pd

    k1_opt, k2_opt, gamma_opt, rho_opt = mejores_params
    gamma_opt=0
    print("\n" + "="*70)
    print("CALIBRACIÓN COMPLETADA. PARÁMETROS ÓPTIMOS ENCONTRADOS:")
    print(f"Mejor Lambda: {mejor_lambda}")
    print(f"k1: {k1_opt:.4f} | k2: {k2_opt:.4f} | Gamma (Localía): {gamma_opt:.4f} | Rho: {rho_opt:.4f}")
    print("="*70 + "\n")
    logger.info(f"k1: {k1_opt:.4f} | k2: {k2_opt:.4f} | Gamma (Localía): {gamma_opt:.4f} | Rho: {rho_opt:.4f}")

    
    # 5. Serialize Model
    # Note: We set HFA to 0 for neutral ground predictions in the final model state
    optimized_params = [k1_opt, k2_opt, gamma_opt, rho_opt] 
    
    
    id_to_team = {v: k for k, v in team_to_id.items()}

# 2. Generamos la lista de CPRs ordenados de menor a mayor ID (0, 1, 2...)
# Usamos .get() por seguridad en caso de que algún ID falte en los datos
    lista_OP = [
        mejores_dict_pg[id_to_team[i]] 
        for i in range(len(team_to_id))
        ]
    lista_DP = [
        mejores_dict_pd[id_to_team[i]] 
        for i in range(len(team_to_id))
        ]
    
    model_obj = DPOPModel(lista_OP,lista_DP,k1_opt, k2_opt, gamma_opt, rho_opt, team_to_id,mejor_ll)
    
    print(model_obj.__class__.__module__)
    with open(MODEL_OUTPUT, 'wb') as f:
        pickle.dump(model_obj, f)
    
    logger.info(f"Model object saved successfully to {MODEL_OUTPUT}")
    
if __name__ == "__main__":
    try:
        run_pipeline()
    finally:
        logging.shutdown()
            