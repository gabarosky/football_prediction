# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 13:46:15 2026

@author: gabri
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import pickle
from collections import Counter
from functools import lru_cache
GROUPS = list("ABCDEFGHIJKL")




THIRD_PLACE_SLOT_COLUMNS = {
    "1A": "slot_1A", "1B": "slot_1B", "1D": "slot_1D", "1E": "slot_1E",
    "1G": "slot_1G", "1I": "slot_1I", "1K": "slot_1K", "1L": "slot_1L",
}

ROUND_OF_32_TEMPLATE = [
    ("2A", "2B"), ("1E", "3"), ("1F", "2C"), ("1C", "2F"),
    ("1I", "3"), ("2E", "2I"), ("1A", "3"), ("1L", "3"),
    ("1D", "3"), ("1G", "3"), ("2K", "2L"), ("1H", "2J"),
    ("1B", "3"), ("1J", "2H"), ("1K", "3"), ("2D", "2G"),
]

R16_PAIRS = [(0, 2), (1, 4), (3, 5), (6, 7), (10, 11), (8, 9), (13, 15), (12, 14)]
QF_PAIRS = [(0, 1), (4, 5), (2, 3), (6, 7)]
SF_PAIRS = [(0, 1), (2, 3)]  
GROUPS = list("ABCDEFGHIJKL")

@lru_cache(maxsize=1)
def third_place_table() -> dict[str, dict[str, str]]:
    """Carga la tabla desde el string embebido en lugar de un archivo."""
    df = pd.read_csv("../data/raw/wc_2026_third_place_table.csv")
    out: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        key = "".join(sorted(str(row["third_groups"])))
        out[key] = {
            slot: str(row[col])
            for slot, col in THIRD_PLACE_SLOT_COLUMNS.items()
        }
    return out

def resolve_slot(slot: str, group_results: dict[str, list], third_assignments: dict[str, str] | None = None) -> str:
    if slot == "3":
        raise ValueError("Generic third-place slot must be resolved by caller")
    if slot.startswith("3"):
        group = slot[1]
        return group_results[group][2]["team"]
    position = int(slot[0]) - 1
    group = slot[1]
    return group_results[group][position]["team"]

def build_round_of_32_bracket(group_results: dict[str, list], best_thirds: list[str]) -> list[str]:
    third_group_by_team = {
        results[2]["team"]: group
        for group, results in group_results.items()
    }
    third_groups = "".join(sorted(third_group_by_team[team] for team in best_thirds))
    assignments = third_place_table()[third_groups]

    bracket: list[str] = []
    for left_slot, right_slot in ROUND_OF_32_TEMPLATE:
        left = resolve_slot(left_slot, group_results)
        if right_slot == "3":
            right_assignment = assignments[left_slot]
            right = resolve_slot(right_assignment, group_results)
        else:
            right = resolve_slot(right_slot, group_results)
        bracket.extend([left, right])
    return bracket

def play_official_knockout_round(winners: list, pairs: list[tuple[int, int]], play_func) -> list:
    return [play_func(winners[a], winners[b]) for a, b in pairs]

def  WorldCupSimulator(model,n: int = 100_000, rng_or_seed=43) -> pd.DataFrame:
    """
    Run n tournament simulations. Returns a DataFrame with columns:
      winner, count, probability
    sorted by probability descending.
    """
    groups_df = load_groups()
    rng = np.random.default_rng(rng_or_seed)

    result = []
    for _ in range(n):
       result.append(simulate_tournament(model, groups_df, rng))
    
    #counts = pd.Series(winners).value_counts().reset_index()
    #counts.columns = ["winner", "count"]
    #counts["probability"] = counts["count"] / n
    return result

def  WorldCupChampSimulator(model,n: int = 100_000, seed: int = 43) -> pd.DataFrame:
    """
    Run n tournament simulations. Returns a DataFrame with columns:
      winner, count, probability
    sorted by probability descending.
    """
    
    rng = np.random.default_rng(seed)
    res = WorldCupSimulator(model,n,rng_or_seed=rng)
    winners = [sim["champ"][0] for sim in res]
    
    # 3. Armamos el DataFrame agrupado que requiere el gráfico
    counts = pd.Series(winners).value_counts().reset_index()
    counts.columns = ["team", "count"]
    counts["prob_champion_%"] = (counts["count"] / n) * 100
    return counts

def simulate_tournament(model, groups_df: pd.DataFrame,  rng) -> str:
    """One complete 2026 WC simulation. Returns all the teams that passed the 
    groups and the level achieved in the tournament"""
    group_results = {}
    for grp in GROUPS:
        teams = groups_df[groups_df["group"] == grp]["team"].tolist()
        group_results[grp] = simulate_group(model, rng, teams)

    thirds  = _best_thirds(group_results)
    bracket = _build_bracket(group_results, thirds)
    lista = _simulate_knockout(model, rng, bracket)
    
    return _simulate_knockout(model, rng, bracket)


def simulate_group(model, rng, teams: list) -> list:
    """
    4-team round-robin using pure ELO win probabilities.
    Scorelines are sampled from Poisson lambdas derived from team attack/defense
    parameters — used solely for goal-difference / goals-for tiebreakers.
    Returns standings sorted by (pts, GD, GF) descending.
    Each entry: {'team': str, 'pts': int, 'gd': int, 'gf': int}
    """
    stats = {t: {"pts": 0, "gd": 0, "gf": 0} for t in teams}
    matches = []
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            a, b = teams[i], teams[j]
            # Outcome from pure ELO probabilities
            ga,gb = model.simulate_match(a,b,rng)["moda"]
             

            stats[a]["gf"] += ga
            stats[b]["gf"] += gb
            stats[a]["gd"] += ga - gb
            stats[b]["gd"] += gb - ga
            if ga>gb:
                stats[a]["pts"] += 3
            elif ga==gb:
                stats[a]["pts"] += 1
                stats[b]["pts"] += 1
            else:
                stats[b]["pts"] += 3
            matches.append((a, b, ga, gb))

    return _rank_group(teams, stats, matches, rng)

def _simulate_knockout(model, rng , bracket: list) -> list:
    """
    Run R32 → R16 → QF → SF → Final.

    R16 cross-pairing: within each section of 4 R32 winners, pair index 0v2, 1v3.
    QF cross-pairing: pair sections (1,2) and (3,4) the same way.
    """
    # R32 — 16 matches
    r32_teams = set(bracket)
    r32w = [_win_knockout(model, rng ,bracket[2*i], bracket[2*i+1]) 
            for i in range(16)]
    r32_eliminated = list(r32_teams - set(r32w))  # Quedaron en 32avos
    
    r16w = play_official_knockout_round(r32w, R16_PAIRS, lambda a, b: _win_knockout(model, rng ,a, b))
    r16_eliminated = list(set(r32w) - set(r16w))
    
    qfw = play_official_knockout_round(r16w, QF_PAIRS, lambda a, b: _win_knockout(model, rng ,a, b))
    qf_eliminated = list(set(r16w) - set(qfw))
    
    # SF
    sfw = play_official_knockout_round(qfw, SF_PAIRS, lambda a, b: _win_knockout(model, rng , a, b))
    sf_eliminated = list(set(qfw) - set(sfw))
    
    champ=_win_knockout(model, rng ,sfw[0], sfw[1])
    runner_up = sfw[1] if champ == sfw[0] else sfw[0]
    return {
        "r32": r32_eliminated,
        "r16": r16_eliminated,
        "r8": qf_eliminated,
        "sfw": sf_eliminated,
        "runner up": [runner_up],
        "champ": [champ]
    }

def _win_knockout(model, rng , a, b):
    ga,gb = model.simulate_match(home_team=a, away_team=b, rng=rng)["moda"]
    if ga==gb:
        elo_a = model.get_rating(a)
        elo_b = model.get_rating(b)
        diff = elo_a-elo_b
        w_a = (1/(1+10**(-diff/400))+.5)/2
        if rng.random()<0.5:
            return a
        else:
            return b
    elif ga<gb:
        return b
    else:
        return a
        


def _rank_group(teams: list, stats: dict, matches: list, rng) -> list:
    """Rank group using FIFA-style goals rules, H2H for exact ties, then random draw."""
    rows = [{"team": t, **stats[t]} for t in teams]
    rows = sorted(rows, key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)
    out = []
    i = 0
    while i < len(rows):
        key = (rows[i]["pts"], rows[i]["gd"], rows[i]["gf"])
        tied = [rows[i]]
        j = i + 1
        while j < len(rows) and (rows[j]["pts"], rows[j]["gd"], rows[j]["gf"]) == key:
            tied.append(rows[j])
            j += 1
        if len(tied) == 1:
            out.extend(tied)
        else:
            tied_teams = {r["team"] for r in tied}
            mini = {t: {"pts": 0, "gd": 0, "gf": 0, "draw": rng.random()} for t in tied_teams}
            for a, b, ga, gb in matches:
                if a not in tied_teams or b not in tied_teams:
                    continue
                mini[a]["gf"] += ga
                mini[b]["gf"] += gb
                mini[a]["gd"] += ga - gb
                mini[b]["gd"] += gb - ga
                if ga > gb:
                    mini[a]["pts"] += 3
                elif ga < gb:
                    mini[b]["pts"] += 3
                else:
                    mini[a]["pts"] += 1; mini[b]["pts"] += 1
            out.extend(sorted(
                tied,
                key=lambda x: (
                    mini[x["team"]]["pts"],
                    mini[x["team"]]["gd"],
                    mini[x["team"]]["gf"],
                    mini[x["team"]]["draw"],
                ),
                reverse=True,
            ))
        i = j
    return out

def _best_thirds(group_results: dict) -> list:
    """Select the 8 best 3rd-place finishers across all 12 groups."""
    thirds = [results[2] for results in group_results.values()]
    return [
        t["team"]
        for t in sorted(thirds, key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)[:8]
    ]



# ── Torunament functions ─────────────────────────────────────────────────────

def load_groups() -> pd.DataFrame:
    """Return wc_2026_groups.csv with normalised team names."""
    df = pd.read_csv(Path("../data/processed/wc_2026_groups.csv"))
    return df

def _build_bracket(group_results: dict, best_thirds: list) -> list:
    """Return official 32-slot R32 bracket using FIFA 2026 Annex-C routing."""
    bracket = build_round_of_32_bracket(group_results, best_thirds)
    assert len(bracket) == 32, f"Expected 32-slot bracket, got {len(bracket)}"
    return bracket
