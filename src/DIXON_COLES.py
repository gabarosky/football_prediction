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

# --- THE MODEL CLASS ---
class DixonColesModel:
    """
    A class to store Dixon-Coles strength and parameters, 
    allowing for easy match predictions.
    """
    def __init__(self, alphas, betas, gamma, rho , team_to_id,xi):
        self.alphas = alphas
        self.betas = betas
        self.gamma = gamma
        self.rho = rho
        self.team_to_id = team_to_id
        self.n_teams = len(alphas)
        self.xi=xi
        self.id_to_team = {v: k for k, v in team_to_id.items()}
        
    @property
    def model(self):
        return self

    def get_teams(self):
        return list(self.team_to_id.keys())
    
    def get_rating(self, team_name):
        team_id = self.team_to_id.get(team_name)
        return self.alphas[team_id]+ self.betas[team_id]
    
    def predict(self, home_team, away_team, avg_goals = 1.3, neutral=True):
        h_idx = self.team_to_id.get(home_team)
        a_idx = self.team_to_id.get(away_team)
        lx = np.exp(self.alphas[h_idx] + self.betas[a_idx] + self.gamma)
        my = np.exp(self.alphas[a_idx] + self.betas[h_idx])
        # score probability matrix
        goals = np.arange(8 + 1)
        px = poisson.pmf(goals, lx)
        py = poisson.pmf(goals, my)
        prob_matrix = np.outer(px, py)
        # rho correction
        prob_matrix[0,0] *= (1 - lx*my*self.rho)
        prob_matrix[0,1] *= (1 + lx*self.rho)
        prob_matrix[1,0] *= (1 + my*self.rho)
        prob_matrix[1,1] *= (1 - self.rho)
        
        prob_matrix /= prob_matrix.sum()
        
        p ={"home_score" :  lx,
            "away_score" :  my,
            "home_win"   :  np.sum(np.tril(prob_matrix, -1)),
            "draw"       :  np.sum(np.diag(prob_matrix)),
            "away_win"   :  np.sum(np.triu(prob_matrix, 1))
            }
        return p
    
    
    def predict_probs(self, home_team, away_team,  max_goals=8, neutral=True):
        """Calculates W/D/L probabilities between two teams."""        
        h_id = self.team_to_id.get(home_team)
        a_id = self.team_to_id.get(away_team)
    
        lx = np.exp(self.alphas[h_id] + self.betas[a_id] + self.gamma)
        my = np.exp(self.alphas[a_id] + self.betas[h_id])
        
        # score probability matrix
        goals = np.arange(max_goals + 1)
        px = poisson.pmf(goals, lx)
        py = poisson.pmf(goals, my)
        prob_matrix = np.outer(px, py)
   
        # rho correction
        prob_matrix[0,0] *= (1 - lx*my*self.rho)
        prob_matrix[0,1] *= (1 + lx*self.rho)
        prob_matrix[1,0] *= (1 + my*self.rho)
        prob_matrix[1,1] *= (1 - self.rho)
   
        prob_matrix /= prob_matrix.sum()
   
        return np.sum(np.tril(prob_matrix, -1)), np.sum(np.diag(prob_matrix)),np.sum(np.triu(prob_matrix, 1))
                
    def simulate_match(self, home_team, away_team, rng,
                                max_goals=8, neutral = True):
        """ Calculate expected goals (lambda) based on Elo difference."""
        h_id = self.team_to_id.get(home_team)
        a_id = self.team_to_id.get(away_team)
    
        lx = np.exp(self.alphas[h_id] + self.betas[a_id] + self.gamma)
        my = np.exp(self.alphas[a_id] + self.betas[h_id])
            
        # score probability matrix
        goals = np.arange(max_goals + 1)
        px = poisson.pmf(goals, lx)
        py = poisson.pmf(goals, my)
        prob_matrix = np.outer(px, py)
   
        # rho correction
        prob_matrix[0,0] *= (1 - lx*my*self.rho)
        prob_matrix[0,1] *= (1 + lx*self.rho)
        prob_matrix[1,0] *= (1 + my*self.rho)
        prob_matrix[1,1] *= (1 - self.rho)
   
        prob_matrix /= prob_matrix.sum()
        
        plain_results = prob_matrix.flatten()
        indices = np.random.choice(
           range(len(plain_results)), 
           size=1, 
           p=plain_results
           )
   
       # convert index to scores
        score_h, score_a = np.divmod(indices, max_goals + 1)
   
       # combine scores
        pairs = list(zip(score_h, score_a))

       # frequency count
        conteo = Counter(pairs)
   
       # obtain the mode
        mode_score, frequency = conteo.most_common(1)[0]
        
        
        # obtain the mode
        
        np.sum(np.tril(prob_matrix, -1)), np.sum(np.diag(prob_matrix)),
        
        e_b = np.ones(max_goals+1) @ prob_matrix @ np.arange(max_goals+1)
        e_a = np.arange(max_goals+1) @prob_matrix @ np.ones(max_goals+1)
        prob_local=np.sum(np.triu(prob_matrix, -1))#resultado.prob_local,
        prob_empate=np.sum(np.diag(prob_matrix))#resultado.prob_empate,
        prob_visita=np.sum(np.triu(prob_matrix, 1))#resultado.prob_visita,
        if prob_local > prob_empate and prob_local > prob_visita:
            resultado_predicho = 'Local'
        elif prob_visita > prob_local and prob_visita > prob_empate:
            resultado_predicho = 'Visitante'
        else:
            resultado_predicho = 'Empate'
        
        val= {
            "prob_matrix"  : prob_matrix, #resultado.matriz,
            "resultado_esp": (e_a,e_b),#resultado.resultado_esperado,
            "moda"         : (mode_score[0], mode_score[1]),#resultado.moda,
            "prob_moda"    : frequency,#resultado.prob_moda,
            "ganador"      : resultado_predicho,#resultado.ganador,
            "prob_local"   : prob_local,
            "prob_empate"  : prob_empate,
            "prob_visita"  : prob_visita
        }
        
        return val

    def simular_n_partidos(self, home_team, away_team, n_sims,
                                max_goals=8, neutral = True):
        """ Calculate expected goals (lambda) based on Elo difference."""
        h_id = self.team_to_id.get(home_team)
        a_id = self.team_to_id.get(away_team)
   
        lx = np.exp(self.alphas[h_id] + self.betas[a_id] + self.gamma)
        my = np.exp(self.alphas[a_id] + self.betas[h_id])
           
       # score probability matrix
        goals = np.arange(max_goals + 1)
        px = poisson.pmf(goals, lx)
        py = poisson.pmf(goals, my)
        prob_matrix = np.outer(px, py)
  
       # rho correction
        prob_matrix[0,0] *= (1 - lx*my*self.rho)
        prob_matrix[0,1] *= (1 + lx*self.rho)
        prob_matrix[1,0] *= (1 + my*self.rho)
        prob_matrix[1,1] *= (1 - self.rho)
  
        prob_matrix /= prob_matrix.sum()
       # Montecarlo: n_simulations
        plain_results = prob_matrix.flatten()
        indices = np.random.choice(
           range(len(plain_results)), 
           size=n_sims, 
           p=plain_results
           )
   
       # convert index to scores
        score_h, score_a = np.divmod(indices, max_goals + 1)
   
        matrix_size = max_goals + 1
        matriz_frecuencias = np.zeros((matrix_size, matrix_size), dtype=int)

# 2. Sumamos 1 en cada coordenada simulada (fuerza bruta vectorial de NumPy)
        np.add.at(matriz_frecuencias, (score_h, score_a), 1)

# [Opcional] Si querés la matriz de probabilidades simuladas para comparar con la original:
        sim_matrix = matriz_frecuencias / n_sims
   
        max_frecuencia = np.max(sim_matrix)

    # 2. Encontrar la fila (goles local) y columna (goles visitante) de ese máximo
        fila_goles_h, col_goles_a = np.unravel_index(np.argmax(matriz_frecuencias), matriz_frecuencias.shape)
       
        np.sum(np.tril(prob_matrix, -1)), np.sum(np.diag(prob_matrix)),
       
        e_b = np.ones(max_goals+1) @ prob_matrix @ np.arange(max_goals+1)
        e_a = np.arange(max_goals+1) @prob_matrix @ np.ones(max_goals+1)
        
        pe_b = np.ones(max_goals+1) @ sim_matrix @ np.arange(max_goals+1)
        pe_a = np.arange(max_goals+1) @sim_matrix @ np.ones(max_goals+1)
        
        print(sim_matrix)
        prob_empate = np.sum(np.diag(sim_matrix))       # diagonal principal (0-0, 1-1, 2-2)
        prob_visita = np.sum(np.triu(sim_matrix, 1))    # triángulo superior (0-1, 0-2, 1-2)
        prob_local  = np.sum(np.tril(sim_matrix, -1))#resultado.prob_visita,
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
           "prob_local"    : prob_local,#resultado.prob_local,
           "prob_empate"   : prob_empate,#resultado.prob_empate,
           "prob_visita"   : prob_visita#resultado.prob_visita,    
           }
        
        return val
    

    def simulate_match_montecarlo(self,home_team,away_team, n_simulations=10000, max_goals=8):    
        h_idx = self.team_to_id.get(home_team)
        a_idx = self.team_to_id.get(away_team)
        
        
        lx = np.exp(self.alphas[h_idx] + self.betas[a_idx] + self.gamma)
        my = np.exp(self.alphas[a_idx] + self.betas[h_idx])
        
        goals = np.arange(max_goals)
        px = poisson.pmf(goals, lx)
        py = poisson.pmf(goals, my)
        prob_matrix = np.outer(px, py)
        
        # Apply rho correction
        prob_matrix[0,0] *= (1 - lx*my*self.rho)
        prob_matrix[0,1] *= (1 + lx*self.rho)
        prob_matrix[1,0] *= (1 + my*self.rho)
        prob_matrix[1,1] *= (1 - self.rho)
        prob_matrix /= prob_matrix.sum()
        
        # Montecarlo: n_simulations
        plain_results = prob_matrix.flatten()
        indices = np.random.choice(
            range(len(plain_results)), 
            size=n_simulations, 
            p=plain_results
        )
        
        # convert index to scores
        score_h, score_a = np.divmod(indices, max_goals)
        
        # combine scores
        pairs = list(zip(score_h, score_a))
    
        # frequency count
        conteo = Counter(pairs)
        
        # obtain the mode
        mode_score, frecuency = conteo.most_common(1)[0]
        # calculate probability score
        probability_score = (frecuency / len(score_h))
    
        
        e_b = np.ones(max_goals) @ prob_matrix @ np.arange(max_goals)
        e_a = np.arange(max_goals) @prob_matrix @ np.ones(max_goals)
        return prob_matrix, (mode_score[0], mode_score[1]), probability_score, (e_a,e_b)    
    
    def predict_score_matrix(self,home_team,away_team, max_goals=8):    
        h_idx = self.team_to_id.get(home_team)
        a_idx = self.team_to_id.get(away_team)
        
        
        lx = np.exp(self.alphas[h_idx] + self.betas[a_idx] + self.gamma)
        my = np.exp(self.alphas[a_idx] + self.betas[h_idx])
        
        goals = np.arange(max_goals)
        px = poisson.pmf(goals, lx)
        py = poisson.pmf(goals, my)
        prob_matrix = np.outer(px, py)
        
        # Apply rho correction
        prob_matrix[0,0] *= (1 - lx*my*self.rho)
        prob_matrix[0,1] *= (1 + lx*self.rho)
        prob_matrix[1,0] *= (1 + my*self.rho)
        prob_matrix[1,1] *= (1 - self.rho)
        prob_matrix /= prob_matrix.sum()
        
        return prob_matrix


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
