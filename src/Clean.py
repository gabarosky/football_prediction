
# -*- coding: utf-8 -*-
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

# 1. Configuración de Logging
LOG_DIR = Path("../logs")
LOG_DIR.mkdir(exist_ok=True)
log_name = f"01_clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE = LOG_DIR / log_name

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, delay=True), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 2. Constantes y Diccionarios (Tu lógica original)
RAW_DATA_PATH = Path("../data/raw")
PROCESSED_DATA_PATH = Path("../data/processed")

NAME_FIXES = {
    'USA': 'United States',
    "Côte d'Ivoire": 'Ivory Coast',
    'IR Iran': 'Iran',
    'Cabo Verde': 'Cape Verde'
}

OFFICIAL_TOURNAMENTS = [
    # worldcup and gobal
   'FIFA World Cup', 'FIFA World Cup qualification', 'Confederations Cup', 
   'FIFA Series', 'FIFA 75th Anniversary Cup',   
   # Europe (UEFA)
   'UEFA Euro', 'UEFA Euro qualification', 'UEFA Nations League',   
   # South America (CONMEBOL)
   'Copa América', 'Copa América qualification', 'CONMEBOL–UEFA Cup of Champions',   
   # North and center America (CONCACAF)
   'Gold Cup', 'Gold Cup qualification', 'CONCACAF Nations League', 
   'CONCACAF Nations League qualification', 'CONCACAF Championship', 
   'CONCACAF Championship qualification', 'UNCAF Cup', 'CFU Caribbean Cup', 
   'CFU Caribbean Cup qualification', 'CCCF Championship', 'NAFC Championship', 
   'NAFU Championship',   
   # Africa (CAF)
   'African Cup of Nations', 'African Cup of Nations qualification', 
   'CECAFA Cup', 'COSAFA Cup', 'COSAFA Cup qualification', 
   'Amílcar Cabral Cup', 'West African Cup', 'UDEAC Cup', 'UNIFFAC Cup',   
   # Asia (AFC)
   'AFC Asian Cup', 'AFC Asian Cup qualification', 'AFC Challenge Cup', 
   'AFC Challenge Cup qualification', 'EAFF Championship', 
   'EAFF Championship qualification', 'WAFF Championship', 'SAFF Cup', 
   'AFF Championship', 'AFF Championship qualification', 'ASEAN Championship', 
   'ASEAN Championship qualification', 'CAFA Nations Cup',   
   # Ocean (OFC)
   'Oceania Nations Cup', 'Oceania Nations Cup qualification', 'Melanesia Cup'
]

def load_data():
    """Load CSVs."""
    try:
        logger.info(f"Loading files from {RAW_DATA_PATH}...")
        results = pd.read_csv(RAW_DATA_PATH / 'results.csv')
        fnames = pd.read_csv(RAW_DATA_PATH / 'former_names.csv')
        teams = pd.read_csv(RAW_DATA_PATH / 'teams.csv')
        fixture = pd.read_csv(RAW_DATA_PATH / 'wc_2026_groups.csv')
        
        return results, fnames, teams, fixture
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        raise

def clean_data(results_df, fnames_df, teams_df, fixture_df):
    """Cleaning and normalization"""
    logger.info("Beggining name filtering and normalization...")
    
    NAME_FIXES = {
    'USA': 'United States',
    'Côte d\'Ivoire': 'Ivory Coast',
    'IR Iran': 'Iran',
    'Cabo Verde': 'Cape Verde',
    'Curacao': 'Curaçao'
    }
    
    # dates and nulls
    results_df['date'] = pd.to_datetime(results_df['date'], format='mixed')
    results_df = results_df.dropna(subset=['home_score']).copy()
    
    # change former names
    former_dict = fnames_df.set_index('former')['current'].to_dict()
        
    results_df['home_team'] = results_df['home_team'].replace(former_dict).str.strip()
    results_df['away_team'] = results_df['away_team'].replace(former_dict).str.strip()
        
    
    teams_df['team_name'] = teams_df['team_name'].str.strip().replace(NAME_FIXES)
    fixture_df['team'] = fixture_df['team'].str.strip().replace(NAME_FIXES)
    
    # Filter official tournaments
    initial_rows = len(results_df)
    results_df = results_df[results_df['tournament'].isin(OFFICIAL_TOURNAMENTS)]
    
    logger.info(f"Filtrado: se mantuvieron {len(results_df)} de {initial_rows} partidos.")
    
    return results_df.sort_values('date').reset_index(drop=True), fixture_df,teams_df['team_name']

def main():
    logger.info("--- DATA PIPELINE STARTED ---")
    try:
        # Load
        df_intl, df_fnames, df_teams, df_fixture = load_data()
        
        # clean
        df_clean, df_fixture_clean,teams = clean_data(df_intl, df_fnames, df_teams, df_fixture)
        
        # folder
        PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
        
        # Save
        df_clean.to_parquet(PROCESSED_DATA_PATH / 'results.parquet')
        df_fixture_clean.to_csv(PROCESSED_DATA_PATH / 'wc_2026_groups.csv', index=False)
        teams.to_csv(PROCESSED_DATA_PATH / 'wc_teams.csv', index=False)
        
        logger.info(f"Pipeline completado. Resultados: {len(df_clean)} filas.")
        
    except Exception as e:
        logger.critical(f"Error fatal en el pipeline: {e}", exc_info=True)
    finally:
        logging.shutdown()

if __name__ == "__main__":
    main()    