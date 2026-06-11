# -*- coding: utf-8 -*-
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import poisson
from scipy.optimize import minimize

# --- LOGGING SETUP ---
LOG_DIR = Path("../logs")
LOG_DIR.mkdir(exist_ok=True)
log_name = f"elo_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(LOG_DIR / log_name, delay=True), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- THE MODEL CLASS ---

class EloModel:
    """
    A class to store ELO ratings and parameters, 
    allowing for easy match predictions.
    """
    def __init__(self, ratings, team_to_id, params):
        self.ratings = ratings
        self.team_to_id = team_to_id
        self.id_to_team = {v: k for k, v in team_to_id.items()}
        self.params = {
            'k_base': params[0],
            'hfa': params[1],
            'draw_margin': params[2]
        }


    # --- MÉTODOS DE INTERFAZ (Lo que agregamos para compatibilidad) ---
    @property
    def model(self):
        return self

    def get_teams(self):
        return list(self.team_to_id.keys())
    
    def get_rating(self, team_name):
        team_id = self.team_to_id.get(team_name)
        return self.ratings[team_id] if team_id is not None else 1500.0

    def predict(self, home_team, away_team, avg_goals = 1.3, neutral=True):
        h_id = self.team_to_id.get(home_team)
        a_id = self.team_to_id.get(away_team)
        
        hfa = 0.0 if neutral else self.params['hfa']
        diff = self.ratings[h_id] - self.ratings[a_id] + hfa
        lambda_h = avg_goals * (10**(diff / 400))
        lambda_a = avg_goals * (10**(-diff / 400))
        margin = self.params['draw_margin']
        p_away = 1 / (1 + 10 ** ((diff + margin) / 400))
        p_home = 1 / (1 + 10 ** (-(diff - margin) / 400))
        p_draw = max(0.0, 1.0 - p_home - p_away)
        p ={"home_score" :  lambda_h,
            "away_score" :  lambda_a,
            "home_win"   :  p_home,
            "draw"       :  p_draw,
            "away_win"   :  p_away
            }
        return p

    def predict_probs(self, home_team, away_team, neutral=True):
        """Calculates W/D/L probabilities between two teams."""
        h_id = self.team_to_id.get(home_team)
        a_id = self.team_to_id.get(away_team)
        
        if h_id is None or a_id is None:
            return 0.33, 0.34, 0.33 # Default for unknown teams

        hfa = 0.0 if neutral else self.params['hfa']
        rat_a= self.get_rating(home_team)
        rat_b= self.get_rating(away_team)
        delta = self.ratings[h_id] + hfa - self.ratings[a_id]
        delta = rat_a + hfa -rat_b
        margin = self.params['draw_margin']
        
        p_away = 1 / (1 + 10 ** ((delta + margin) / 400))
        p_home = 1 / (1 + 10 ** (-(delta - margin) / 400))
        p_draw = max(0.0, 1.0 - p_home - p_away)
        return p_home, p_draw, p_away
    
    def simulate_match(self,home_team, away_team, rng , avg_goals=1.3):
        # Calculate expected goals (lambda) based on Elo difference
        elo_a = self.get_rating(home_team)
        elo_b = self.get_rating(away_team)
        diff = elo_a - elo_b
        lambda_a = avg_goals * (1/(1+10**(-diff / 400)))
        lambda_b = avg_goals * (1/(1+10**(diff / 400)))
    
        # Generate random goals
        val= {
            "moda"         : (rng.poisson(lambda_a,1), rng.poisson(lambda_b,1))#resultado.moda,
            
        }
        return val
    
    
    
    
    def simular_n_partidos(self, home_team, away_team, n_sims,
                                max_goals=8, neutral = True):
        """ Calculate expected goals (lambda) based on Elo difference."""
        max_goals = 8
        elo_a = self.get_rating(home_team)
        elo_b = self.get_rating(away_team)
        diff = elo_a - elo_b
        avg_goals=1.3
        rng = np.random.default_rng()
        lambda_a = avg_goals * (1 / (1 + 10**(-diff / 400)))
        lambda_b = avg_goals * (1 / (1 + 10**(diff / 400)))
    
        sample_size = max(10 * n_sims, 10000)
        goals_a = rng.poisson(lambda_a, sample_size)
        goals_b = rng.poisson(lambda_b, sample_size)
        # goals_a = rng.poisson(lambda_a, 1)
        # goals_b = rng.poisson(lambda_b, 1)
        A, _, _ = np.histogram2d(
            goals_a, goals_b, 
            bins=(np.arange(max_goals + 2), np.arange(max_goals + 2))
            )
        
     
        
        prob_matrix = A/A.sum()
       
       # Montecarlo: n_simulations
        plain_results = prob_matrix.flatten()
        indices = np.random.choice(
           range(len(plain_results)), 
           size=n_sims, 
           p=plain_results
           )
   
       # convert index to scores
        matrix_size = max_goals +1
        score_h, score_a = np.divmod(indices, matrix_size)
   
        
        matriz_frecuencias = np.zeros((matrix_size, matrix_size), dtype=int)

# 2. Sumamos 1 en cada coordenada simulada (fuerza bruta vectorial de NumPy)
        np.add.at(matriz_frecuencias, (score_h, score_a), 1)

# [Opcional] Si querés la matriz de probabilidades simuladas para comparar con la original:
        sim_matrix = matriz_frecuencias / matriz_frecuencias.sum()
   
        max_frecuencia = np.max(sim_matrix)

    # 2. Encontrar la fila (goles local) y columna (goles visitante) de ese máximo
        fila_goles_h, col_goles_a = np.unravel_index(np.argmax(matriz_frecuencias), matriz_frecuencias.shape)
       
        np.sum(np.tril(prob_matrix, -1)), np.sum(np.diag(prob_matrix)),
        
        e_b = np.ones(max_goals+1) @ prob_matrix @ np.arange(max_goals+1)
        e_a = np.arange(max_goals+1) @prob_matrix @ np.ones(max_goals+1)
        
        pe_b = np.ones(max_goals+1) @ sim_matrix @ np.arange(max_goals+1)
        pe_a = np.arange(max_goals+1) @sim_matrix @ np.ones(max_goals+1)
        
        
        prob_local=np.sum(np.triu(sim_matrix, -1))#resultado.prob_local,
        prob_empate=np.sum(np.diag(sim_matrix))#resultado.prob_empate,
        prob_visita=np.sum(np.triu(sim_matrix, 1))#resultado.prob_visita,
        if prob_local > prob_empate and prob_local > prob_visita:
            resultado_predicho = 'Local'
        elif prob_visita > prob_local and prob_visita > prob_empate:
            resultado_predicho = 'Visitante'
        else:
            resultado_predicho = 'Empate'
        
        val= {
           "prob_matrix"   : prob_matrix, #resultado.matriz,
           "resultado_esp" : (e_a,e_b),#resultado.resultado_esperado,
           "resultado_pred": (pe_a,pe_b),
           "moda"          : (fila_goles_h, col_goles_a ),#resultado.moda,
           "prob_moda"     : max_frecuencia,#resultado.prob_moda,
           "ganador"       : resultado_predicho,#resultado.ganador,
           "prob_local"    : np.sum(np.triu(prob_matrix, -1)),#resultado.prob_local,
           "prob_empate"   : np.sum(np.diag(prob_matrix)),#resultado.prob_empate,
           "prob_visita"   : np.sum(np.triu(prob_matrix, 1)),#resultado.prob_visita,
        }
        
        return val
    

    def simulate_match_montecarlo(self,home_team, away_team,rng ,n_times, avg_goals=1.3):
        max_goals = 8
        elo_a = self.get_rating(home_team)
        elo_b = self.get_rating(away_team)
        diff = elo_a - elo_b
        lambda_a = avg_goals * (1 / (1 + 10**(-diff / 400)))
        lambda_b = avg_goals * (1 / (1 + 10**(diff / 400)))
    
        goals_a = rng.poisson(lambda_a, n_times)
        goals_b = rng.poisson(lambda_b, n_times)
    
        A, _, _ = np.histogram2d(
            goals_a, goals_b, 
            bins=(np.arange(max_goals + 1), np.arange(max_goals + 1))
            )
        probs = A/A.sum()
        e_b = np.ones(max_goals) @ probs @ np.arange(max_goals)
        e_a = np.arange(max_goals) @ probs @ np.ones(max_goals)
    
        a_score, b_score = np.unravel_index(np.argmax(probs), probs.shape)

        return probs, (a_score, b_score), np.max(probs), (e_a,e_b)

    def predict_score_matrix(self,home_team, away_team, max_goals=8):
        rng =np.random.default_rng()
        n_times=10000
        avg_goals=1.3
        max_goals = 8
        elo_a = self.get_rating(home_team)
        elo_b = self.get_rating(away_team)
        diff = elo_a - elo_b
        lambda_a = avg_goals * (1 / (1 + 10**(-diff / 400)))
        lambda_b = avg_goals * (1 / (1 + 10**(diff / 400)))
   
        goals_a = rng.poisson(lambda_a, n_times)
        goals_b = rng.poisson(lambda_b, n_times)
   
        A, _, _ = np.histogram2d(
            goals_a, goals_b, 
            bins=(np.arange(max_goals + 1), np.arange(max_goals + 1))
            )
        probs = A/A.sum()
        
        return probs
    
# --- TRAINING LOGIC ---



class ModelLoader:
    @staticmethod
    def load(path, verbose=False):
        # Maneja solo archivos pickle, como es tu caso
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        # Inyectamos la estructura que el notebook necesita si no la tiene
        if not hasattr(model, 'model'):
            model.model = model
        return model


        
"""
import pickle

# Cargar el objeto completo
with open('../saved_models/elo_model_v1.pkl', 'rb') as f:
    model = pickle.load(f)

# El objeto ya sabe qué hacer:
p_win, p_draw, p_loss = model.predict_probs("Argentina", "France")
print(f"Probabilidad de que Argentina gane: {p_win:.2%}")

# También puedes consultar ratings directamente
print(f"Rating de Argentina: {model.get_rating('Argentina'):.1f}")
"""        