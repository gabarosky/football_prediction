# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 09:33:22 2026

@author: gabri
"""

# -*- coding: utf-8 -*-
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import poisson
from scipy.optimize import minimize
from collections import Counter
from src.DIXON_COLES import DixonColesModel
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Ahora sí podrás usarla aquí sin errores:
INPUT_FILE = os.path.join(SCRIPT_DIR, "..", "data", "processed", "results.parquet")

# --- LOGGING SETUP ---
LOG_DIR = Path("../logs")
LOG_DIR.mkdir(exist_ok=True)
log_name = f"Dixon-Coles_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(LOG_DIR / log_name, delay=True), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- TRAINING LOGIC ---


def dc_neg_log_like_with_grad(params, h_idx, a_idx, x, y, is_home, t_diff, n_teams, xi=0.004, l2_reg=0.001):
    # unpacking parameters
    alphas = params[:n_teams]
    betas = params[n_teams:2*n_teams]
    gamma = params[2*n_teams]
    rho = params[2*n_teams + 1]
    
    # time weight
    weights = np.exp(-xi * t_diff)
    
    # offensive and deffensive intensities (λ and μ)
    lambda_x = np.exp(alphas[h_idx] + betas[a_idx] + gamma * is_home)
    mu_y = np.exp(alphas[a_idx] + betas[h_idx])
    
    # correction for low scores (tau)
    tau = np.ones_like(lambda_x)
    m00 = (x == 0) & (y == 0)
    m10 = (x == 1) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m11 = (x == 1) & (y == 1)
    
    tau[m00] = 1.0 - lambda_x[m00] * mu_y[m00] * rho
    tau[m10] = 1.0 + lambda_x[m10] * rho
    tau[m01] = 1.0 + mu_y[m01] * rho
    tau[m11] = 1.0 - rho
    
    # ovoiding nummerical issues
    eps = 1e-10
    lambda_x = np.clip(lambda_x, eps, 1e10)
    mu_y = np.clip(mu_y, eps, 1e10)
    tau = np.clip(tau, eps, None)
    
    # objective function Negative Log-Likelihood (NLL)
    # Poisson: lambda - x*log(lambda) | Dixon-Coles: -log(tau)
    poisson_part = lambda_x + mu_y - x * np.log(lambda_x) - y * np.log(mu_y)
    nll = np.sum(weights * (poisson_part - np.log(tau)))
    
    # L2 regularization
    nll += l2_reg * (np.sum(alphas**2) + np.sum(betas**2))
    
    # gradient function
    A = np.zeros_like(lambda_x)
    B = np.zeros_like(lambda_x)
    
    A[m00], B[m00] = -mu_y[m00] * rho / tau[m00], -lambda_x[m00] * rho / tau[m00]
    A[m10] = rho / tau[m10]
    B[m01] = rho / tau[m01]
    
    # partial derivatives
    d_lambda = weights * (lambda_x - x - A * lambda_x)
    d_mu = weights * (mu_y - y - B * mu_y)
    
    # group by teams (Alphas y Betas)
    grad_alphas = np.bincount(h_idx, weights=d_lambda, minlength=n_teams) + \
                  np.bincount(a_idx, weights=d_mu, minlength=n_teams)
                  
    grad_betas = np.bincount(a_idx, weights=d_lambda, minlength=n_teams) + \
                 np.bincount(h_idx, weights=d_mu, minlength=n_teams)
    
    # gamma and rho derivative
    grad_gamma = np.sum(d_lambda * is_home)
    
    grad_rho_part = np.zeros_like(lambda_x)
    grad_rho_part[m00] = -lambda_x[m00] * mu_y[m00] / tau[m00]
    grad_rho_part[m10] = lambda_x[m10] / tau[m10]
    grad_rho_part[m01] = mu_y[m01] / tau[m01]
    grad_rho_part[m11] = -1.0 / tau[m11]
    grad_rho = -np.sum(weights * grad_rho_part)
    
    # L2 regularization 
    grad_alphas += 2 * l2_reg * alphas
    grad_betas += 2 * l2_reg * betas
    
    
    grad = np.concatenate([grad_alphas, grad_betas, [grad_gamma, grad_rho]])
    
    return nll, grad
    
def run_pipeline():
    # 1. Paths
    INPUT_FILE = os.path.join(SCRIPT_DIR, "..", "data", "processed", "results.parquet")
    MODEL_OUTPUT = Path(os.path.join(SCRIPT_DIR, "..", "saved_models", "dixoncoles_model_v1.pkl"))
    MODEL_OUTPUT.parent.mkdir(exist_ok=True)

    # 2. Load and Prepare
    logger.info("Loading processed data...")
    df = pd.read_parquet(INPUT_FILE)
    unique_teams = sorted(list(set(df['home_team']).union(set(df['away_team']))))
    team_to_id = {team: i for i, team in enumerate(unique_teams)}
    n_teams = len(unique_teams)
    
    h_idx_all = df['home_team'].map(team_to_id).values
    a_idx_all = df['away_team'].map(team_to_id).values
    h_goals_all = df['home_score'].values
    a_goals_all = df['away_score'].values
    dates_all = df['date'].values
        
    params_guess = np.zeros(2 * n_teams + 2)
    params_guess[2*n_teams] = 0.0
    params_guess[2*n_teams + 1] = 0.25 
        
    bounds = ([(None, None)] * (2 * n_teams + 1)) + [(-0.15, 0.15)]
    
    xi=0.004
    i= 21715
    h_idx_train = h_idx_all[:i]
    a_idx_train = a_idx_all[:i]
    x_train = h_goals_all[:i]
    y_train = a_goals_all[:i]
    is_home_train = np.ones(len(x_train)) 
                
    # time factor
    last_date = pd.to_datetime(dates_all[i-1])
    past_dates = pd.to_datetime(dates_all[:i])
    # t_diff in weeks
    t_diff = (last_date - past_dates).days / 7.0
    
    res = minimize(
                    fun=dc_neg_log_like_with_grad,
                    x0=params_guess,
                    args=(h_idx_train, a_idx_train, x_train, y_train, is_home_train, t_diff, n_teams, xi, 0.001),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=bounds
                )
    
    
    params = res.x
    
    logger.info(f"Best Params calculated.")
    alphas = params[0:n_teams]
    betas = params[n_teams:2*n_teams]
    gamma = params[2*n_teams]
    rho = params[2*n_teams + 1]
    
    h_id = team_to_id.get("Qatar")
    a_id = team_to_id.get("Ecuador")

    lx = np.exp(alphas[h_id] + betas[a_id] )
    my = np.exp(alphas[a_id] + betas[h_id])
    gamma=0.0
    
    model_obj = DixonColesModel(alphas, betas, gamma, rho, team_to_id,xi)
    
    with open(MODEL_OUTPUT, 'wb') as f:
        pickle.dump(model_obj, f)
    
    logger.info(f"Model object saved successfully to {MODEL_OUTPUT}")    

    
if __name__ == "__main__":
    try:
        run_pipeline()
    finally:
        logging.shutdown()
            
"""
with open('../saved_models/dixoncoles_model_v1.pkl', 'rb') as f:
    model = pickle.load(f)
    """