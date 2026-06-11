# -*- coding: utf-8 -*-
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.optimize import minimize
from src.ELO import EloModel
import os
import sys
# --- LOGGING SETUP ---

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOG_DIR = Path("../logs")
LOG_DIR.mkdir(exist_ok=True)
log_name = f"elo_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(LOG_DIR / log_name, delay=True), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)



def get_k_multiplier(n):
    n = abs(n)
    if n <= 1: return 1.0
    if n == 2: return 1.5
    if n == 3: return 1.75
    return 1.75 + (n - 3) / 8.0

def objective_function(params, data_array, num_teams, burn_in):
    k_base, hfa, draw_margin = params
    ratings = np.full(num_teams, 1500.0)
    total_rps = 0.0
    count = 0
    
    for i in range(len(data_array)):
        h_id, a_id, h_s, a_s = data_array[i]
        delta = ratings[h_id] + hfa - ratings[a_id]
        
        p_away = 1 / (1 + 10 ** ((delta + draw_margin) / 400))
        p_home = 1 / (1 + 10 ** (-(delta - draw_margin) / 400))
        p_draw = max(0.0, 1.0 - p_home - p_away)
        
        actual = 0 if h_s > a_s else 1 if h_s == a_s else 2
        
        if i >= burn_in:
            e_cum_0 = 1.0 if actual == 0 else 0.0
            e_cum_1 = 1.0 if actual <= 1 else 0.0
            total_rps += ((p_home - e_cum_0)**2 + (p_home + p_draw - e_cum_1)**2) / 2.0
            count += 1
        
        # Update ratings
        e = 1.0 if actual == 0 else 0.5 if actual == 1 else 0.0
        e_home = 1 / (1 + 10 ** (-delta / 400))
        shift = (k_base * get_k_multiplier(h_s - a_s)) * (e - e_home)
        ratings[h_id] += shift
        ratings[a_id] -= shift
        
    return total_rps / count if count > 0 else 0


def run_pipeline():
    # 1. Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Correctly goes up one level, then into data/processed/
    INPUT_FILE = os.path.join(SCRIPT_DIR, "..", "data", "processed", "results.parquet")
    MODEL_OUTPUT = Path(os.path.join(SCRIPT_DIR, "..", "saved_models", "elo_model_v1.pkl"))
    MODEL_OUTPUT.parent.mkdir(exist_ok=True)

    # 2. Load and Prepare
    logger.info("Loading processed data...")
    df = pd.read_parquet(INPUT_FILE)
    all_teams = sorted(list(set(df['home_team']).union(set(df['away_team']))))
    team_to_id = {team: i for i, team in enumerate(all_teams)}
    
    df['h_id'] = df['home_team'].map(team_to_id)
    df['a_id'] = df['away_team'].map(team_to_id)
    
    WARMING_ELO = 18935
    TRAIN_END = 21715
    data_array = df.iloc[:TRAIN_END][['h_id', 'a_id', 'home_score', 'away_score']].values.astype(int)
    
    # 3. Optimize
    logger.info("Optimizing ELO parameters (L-BFGS-B)...")
    res = minimize(
        objective_function, [20.0, 40.0, 90.0], 
        args=(data_array, len(all_teams), WARMING_ELO),
        bounds=[(5, 80), (0, 150), (10, 200)], method='L-BFGS-B'
    )
    
    best_params = res.x
    logger.info(f"Best Params: K={best_params[0]:.2f}, HFA={best_params[1]:.2f}, Margin={best_params[2]:.2f}")

    # 4. Final Pass to get Final Ratings
    # Run the ELO logic one last time with optimal parameters
    logger.info("Calculating final team ratings...")
    final_ratings = np.full(len(all_teams), 1500.0)
    for h_id, a_id, h_s, a_s in data_array:
        delta = final_ratings[h_id] + best_params[1] - final_ratings[a_id]
        actual = 0 if h_s > a_s else 1 if h_s == a_s else 2
        e = 1.0 if actual == 0 else 0.5 if actual == 1 else 0.0
        e_home = 1 / (1 + 10 ** (-delta / 400))
        shift = (best_params[0] * get_k_multiplier(h_s - a_s)) * (e - e_home)
        final_ratings[h_id] += shift
        final_ratings[a_id] -= shift

    # 5. Serialize Model
    # Note: We set HFA to 0 for neutral ground predictions in the final model state
    optimized_params = [best_params[0],best_params[1], best_params[2]] 
    model_obj = EloModel(final_ratings, team_to_id, optimized_params)
    
    print(model_obj.__class__.__module__)
    with open(MODEL_OUTPUT, 'wb') as f:
        pickle.dump(model_obj, f)
    
    logger.info(f"Model object saved successfully to {MODEL_OUTPUT}")
    
if __name__ == "__main__":
    try:
        run_pipeline()
    finally:
        logging.shutdown()
            