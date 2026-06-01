# ⚽ FIFA World Cup 2026 — Prediction & Simulation Engine

> A data science project exploring multiple statistical and graph-based approaches
> to model, analyze, and simulate the 2026 FIFA World Cup.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Status](#project-status)
- [Repository Structure](#repository-structure)
- [Data Sources](#data-sources)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Prediction Models](#prediction-models)
  - [ELO Rating System](#1-elo-rating-system)
  - [Dixon-Coles Model](#2-dixon-coles-model)
  - [Own Models](#3-own-models)
- [Simulation Engine](#simulation-engine)
- [Streamlit App](#streamlit-app-coming-soon)
- [Installation](#installation)
- [Usage](#usage)
- [Results & Comparisons](#results--comparisons)
- [Roadmap](#roadmap)
- [References](#references)

---

## Overview

This project builds a modular prediction and simulation engine for the **2026 FIFA World Cup** (USA, Canada & Mexico), featuring:

- A thorough **Exploratory Data Analysis** of the historical performance of the 48 qualified teams.
- **Multiple independent prediction models** — from classical rating systems to advanced statistical and graph-based approaches — each independently trainable, saveable, and loadable.
- A **Monte Carlo simulation engine** capable of running thousands of full-tournament simulations and producing championship probability distributions.
- A future **interactive Streamlit app** where users can pick any model, simulate the full tournament or a single match, and explore the results visually.

The codebase is deliberately designed around a common interface (`BaseFootballModel`) so that every model is a drop-in replacement for any other — adding a new model requires touching only two files.

---

## Project Status

| Component | Status | Notebook |
|---|---|---|
| Data collection & cleaning | ✅ Done | `00_data_collection.ipynb` |
| Exploratory Data Analysis | ✅ Done | `01_EDA.ipynb` |
| ELO Rating System | 🔄 In progress | `02_ELO.ipynb` |
| Dixon-Coles Model | 🔄 In progress | `03_Dixon_Coles.ipynb` |
| Own Model v1 | 🔲 Planned | `04_Own_v1.ipynb` |
| Own Model v2 | 🔲 Planned | `05_Own_v2.ipynb` |
| Model comparison & evaluation | 🔲 Planned | `06_Model_Comparison.ipynb` |
| Streamlit app | 🔲 Planned | `app/` |

---

## Repository Structure

```
wc2026-prediction/
│
├── data/
│   ├── raw/                        # Original downloaded datasets
│   │   ├── former_names.csv
│   │   ├── teams.csv
│   │   └── results.csv
│   ├── processed/                  # Cleaned, feature-engineered data
│   │   ├── results.parquet
│   │   └── wc2026_groups.csv
│   └── results/                    # Model outputs & simulation results
│       ├── elo_ratings.csv
│       ├── dixon_coles_strengths.csv
│       └── monte_carlo_summary.csv
│
├── notebooks/
│   ├── 00_data_collection.ipynb
│   ├── 01_EDA.ipynb                 ✅
│   ├── 02_ELO.ipynb                 🔄
│   ├── 03_Dixon_Coles.ipynb         🔄
│   ├── 04_Own_v1.ipynb              🔄
│   ├── 05_Own.ipynb                 🔲
│   └── 06_Model_Comparison.ipynb    🔲
│
├── football_predictor/             # Core Python package
│   ├── base_model.py               # Abstract interface — the scalability contract
│   ├── trainer.py                  # Train / save / load any registered model
│   ├── predictor.py                # Single-match forecasts with rich output
│   ├── simulator.py                # Full tournament Monte Carlo engine
│   ├── wc2026_fixture.py           # Official WC 2026 groups & fixture
│   └── models/
│       ├── dixon_coles.py          ✅
│       ├── elo.py                  🔄
│       ├── pagerank_v1.py          🔲
│       └── pagerank_v2.py          🔲
│
├── saved_models/                   # Serialized trained models (.json.gz)
│
├── app/                            # Streamlit application (coming soon)
│   ├── app.py
│   └── components/
│
├── requirements.txt
└── README.md
```

---

## Data Sources

| Dataset | Description | Source |
|---|---|---|
| International match results | ~45,000 matches since 1872 | [Kaggle — International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) |
| WC 2026 groups | Official draw, December 2024 | FIFA |

<!-- **Training window:** matches from **2018 to present**, weighted by recency. Competitive matches (World Cup, Copa América, UEFA Nations League, etc.) are given higher weight than friendlies. --->

---

## Exploratory Data Analysis

> 📓 Notebook: `01_EDA.ipynb`

The EDA characterizes each of the 48 qualified nations inside the context of their assigned group, covering both historical form and structural patterns in international football scoring.
<!--
### Key analyses

**Group-level overview**
- Distribution of team strength (ELO / FIFA ranking) per group — identifying "groups of death" vs balanced groups.
- Geographic and confederation breakdown of WC 2026 participants.

**Scoring & match dynamics**
- Distribution of goals scored and conceded per team (home vs. away vs. neutral).
- Historical clean sheet and high-scoring game rates by team.
- Head-to-head record matrix for teams within the same group.

**Temporal trends**
- Team form over the last 2 years (rolling win rate, goals/game).
- Performance decay analysis: do teams peak at tournaments?

**Structural patterns**
- Home advantage magnitude across confederations.
- Low-score game frequency (the 0-0, 1-0, 1-1 cluster) — motivating the Dixon-Coles correction.
- Overdispersion test: do goal counts follow Poisson, or do we need corrections?
--->
---

## Prediction Models
<!--
All models share a common interface defined in `football_predictor/base_model.py`. They are fully interchangeable: any model can be passed to `MatchPredictor` or `WorldCupSimulator` without any other code change.
-->
---

### 1. ELO Rating System

> 📓 Notebook: `02_ELO.ipynb` | 🟢 Status: Done

A classical chess-derived rating system adapted for international football.

**How it works**

Each team carries a rating $R$. After every match, ratings are updated based on the expected vs. actual outcome:

$$ R'_{\text{home}} = R_{\text{home}} + K \cdot (S - E) $$


$$R_{\text{new}} = R_{\text{old}} + K \times (W - W_e)$$

where $W$ is the result ($1$ for local win, $0.5$ for a draw and $0$ for visitor win) and $W_e = \frac{1}{10^{(R_{\text{away}} - R_{\text{home}} - \delta) / 400}}$ is the expected score and $\delta$ is the home advantage offset.

**Key design choices**
- Starting ratings are estimated with matches previous to FIFA WC 2018.
- Parameteres otimized for period FIFA WC 2018 to FICA WC 2022.
- In WC the home advantage is fixed to 0.
- Match outcome is encoded as 1 / 0.5 / 0 (win / draw / loss).

**Output:** a probability vector $(p_H, p_D, p_A)$ derived from the ELO difference via logistic regression calibrated on historical outcomes.

**Strengths:** simple, interpretable, fast. Works well for win/loss prediction.

**Limitations:** does not model goal counts; cannot produce score matrices; provides no information for Over/Under markets.

---

### 2. Dixon-Coles Model

> 📓 Notebook: `03_Dixon_Coles.ipynb` | 🟢 Status: Done

A bivariate Poisson regression model that explicitly models the number of goals scored by each team.

**How it works**

Goals are modelled as Poisson random variables:

$$X_H \sim \text{Poisson}(\lambda), \quad X_A \sim \text{Poisson}(\mu)$$

$$\lambda = \exp(\alpha_i + \beta_j + \gamma), \quad \mu = \exp(\alpha_j + \beta_i)$$

where $\alpha_i$ is team $i$'s attack strength, $\beta_j$ is team $j$'s defense weakness, and $\gamma$ captures home advantage.

The model adds a **low-score correction** (the Dixon-Coles $\tau$ adjustment) that fixes the underestimation of 0-0, 1-0, 0-1 and 1-1 scorelines that plain Poisson produces.

**Temporal weighting**

Matches are weighted by recency using an exponential decay:

$$w(t) = e^{-\xi \cdot t_{\text{weeks}}}$$

The $\xi$ hyperparameter controls how fast old matches become irrelevant.

**Strengths:** produces a full score probability matrix; naturally yields 1X2 probabilities, expected goals, and Over/Under probabilities. Well-suited for Monte Carlo.

**Limitations:** assumes independence between the two teams' goal processes (mitigated by $\tau$); doesn't model red cards, injuries, or tournament pressure.

---

### 3. Own Models

<!--
> 📓 `04_PageRank_v1.ipynb` 🔄 | `05_PageRank_v2.ipynb` 🔲

Treats international football as a **directed weighted graph** where an edge $i \rightarrow j$ exists if team $i$ scored against team $j$. PageRank's stationary distribution then gives a global strength measure not available from pairwise win/loss records alone.

#### 3a. PageRank v1 — Goal-Weighted Directed Graph

**Graph construction**
- Nodes: national teams.
- Edge $i \rightarrow j$ with weight = total goals scored by $i$ against $j$ (recency-weighted).
- Intuition: scoring against a strong team (one that itself scores against strong opponents) should count more.

**From PageRank score to match probability**
- The ratio $\text{PR}(i) / \text{PR}(j)$ is passed through a calibrated logistic function to produce win probabilities.
- A separate Poisson regression is fitted to predict expected goals, using the PageRank score as a covariate.

#### 3b. PageRank v2 — *(design TBD)*

Possible directions under consideration:
- Personalized PageRank (restart vector biased towards recent opponents).
- Separate offensive and defensive PageRank scores (analogous to HITS hubs/authorities).
- Incorporating margin of victory into edge weights with a diminishing-returns function to prevent blowout inflation.
-->
---

## Simulation Engine
<!--
> 📦 `football_predictor/simulator.py`

The `WorldCupSimulator` class runs Monte Carlo simulations of the full 2026 World Cup format:

- **48 teams** across **12 groups of 4**
- Top 2 from each group + 8 best third-placed teams advance to Round of 32
- Full knockout bracket through Round of 16 → Quarterfinals → Semifinals → Final
- Third-place playoff simulated separately
- Knockout ties resolved via simulated penalty shootout (50/50 by default; configurable)

### Running simulations

```python
from football_predictor import ModelTrainer, WorldCupSimulator
from football_predictor.wc2026_fixture import get_groups_fixture

model = ModelTrainer.load("saved_models/dixon_coles_wc2026.json.gz")
sim   = WorldCupSimulator(model, get_groups_fixture())

# Single simulation with printed bracket
result = sim.simulate_tournament(verbose=True)

# Monte Carlo — championship probabilities
df_probs = sim.monte_carlo(n_sim=50_000)
```

### Example output *(illustrative — real results pending full model training)*

| Team | P(Groups) | P(QF) | P(SF) | P(Final) | P(Champion) |
|---|---|---|---|---|---|
| Argentina | — | — | — | — | — |
| France | — | — | — | — | — |
| Brazil | — | — | — | — | — |
| England | — | — | — | — | — |
| ... | | | | | |

*Table will be populated once models are trained on the full dataset.*
-->
---

## Streamlit App *(coming soon)*

> 📦 `app/` | 🔲 Status: Planned

An interactive web application where users can:

**Simulate the full tournament**
- Select any trained model (ELO, Dixon-Coles, PageRank v1/v2).
- Run N Monte Carlo simulations (slider: 1,000 – 100,000).
- Visualize championship probability distributions, bracket heatmaps, and group stage pass-through rates.

**Simulate a single match**
- Choose home and away team from dropdowns.
- Select model.
- See the probability of each outcome, expected goals, and the most likely scorelines as a heatmap.

**Compare models**
- Side-by-side view of probability estimates from multiple models for the same match or tournament stage.

---

## Installation
<!--
```bash
git clone https://github.com/your-username/wc2026-prediction.git
cd wc2026-prediction

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**Core dependencies**

```
numpy
pandas
scipy
scikit-learn
networkx          # PageRank models
matplotlib
seaborn
jupyter
```
-->
---

## Usage
<!--
### Train and save a model

```python
import pandas as pd
from football_predictor import ModelTrainer

df = pd.read_csv("data/processed/matches_clean.csv")

trainer = ModelTrainer(
    model_name="dixon_coles",
    save_dir="saved_models",
    model_kwargs={"xi": 0.005},
)
trainer.train(df)
trainer.save("dixon_coles_wc2026")
```

### Load and predict a match

```python
from football_predictor import ModelTrainer, MatchPredictor

model     = ModelTrainer.load("saved_models/dixon_coles_wc2026.json.gz")
predictor = MatchPredictor(model)

predictor.print_report("Argentina", "France")
```

### Run a full Monte Carlo simulation

```python
from football_predictor import WorldCupSimulator
from football_predictor.wc2026_fixture import get_groups_fixture

sim      = WorldCupSimulator(model, get_groups_fixture())
df_probs = sim.monte_carlo(n_sim=50_000)

print(df_probs.head(10))
```

### Adding a new model

All models share a common interface. To add your own:

1. Create `football_predictor/models/my_model.py` inheriting from `BaseFootballModel`.
2. Set `MODEL_NAME = "my_model"` on the class.
3. Implement the 7 required methods (`fit`, `predict_match_probs`, `simulate_match_score`, `get_params`, `set_params`, `get_teams`, `is_fitted`).
4. Register it in `ModelTrainer.REGISTRY` in `trainer.py`.

See `football_predictor/nuevo_modelo_template.py` for a fully annotated template and compatibility checklist.
-->
---

## Results & Comparisons

> *This section will be updated as models are trained and evaluated.*

### Evaluation methodology

Models are evaluated on held-out matches (2024–2025) using:
<!--
| Metric | Description |
|---|---|
| **RPS** (Ranked Probability Score) | Measures calibration across all three outcomes. Lower is better. Naive baseline: ~0.250. |
| **Brier Score** | Mean squared error of probability estimates. |
| **Log Loss** | Penalises overconfident wrong predictions heavily. |
| **Accuracy** | Fraction of correctly predicted outcomes (1X2). |

### Model comparison table

| Model | RPS | Brier | Log Loss | Accuracy |
|---|---|---|---|---|
| Naive baseline (1/3 each) | 0.250 | — | — | — |
| ELO | — | — | — | — |
| Dixon-Coles | — | — | — | — |
| PageRank v1 | — | — | — | — |
| PageRank v2 | — | — | — | — |
-->
---

## Roadmap

- [x] Data collection and cleaning pipeline
- [x] Exploratory Data Analysis (EDA)
- [ ] ELO model — training, evaluation, serialization
- [ ] Dixon-Coles model — training, evaluation, serialization
- [ ] Modular prediction engine (`BaseFootballModel` interface)
- [ ] Monte Carlo tournament simulator
- [ ] ELO integrated into the `football_predictor` package
- [ ] Own v1 model
- [ ] Own v2 model 
- [ ] Model evaluation notebook with RPS / Brier comparison
- [ ] Streamlit app — match predictor
- [ ] Streamlit app — tournament simulator
- [ ] Streamlit app — model comparison view
- [ ] Docker deployment

---

## References

- Dixon, M.J. & Coles, S.G. (1997). *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*. Applied Statistics, 46(2), 265–280.
- Elo, A.E. (1978). *The Rating of Chessplayers, Past and Present*. Arco Publishing.
<!-- - Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). *The PageRank Citation Ranking: Bringing Order to the Web*. Stanford InfoLab. -->
- Maher, M.J. (1982). *Modelling Association Football Scores*. Statistica Neerlandica, 36(3), 109–118.
- Constantinou, A.C. & Fenton, N.E. (2013). *Determining the level of ability of football teams by dynamic ratings based on the relative discrepancies in scores between adversaries*. Journal of Quantitative Analysis in Sports.

---

<p align="center">
  Made with ⚽ and 🐍 &nbsp;|&nbsp; FIFA World Cup 2026
</p>