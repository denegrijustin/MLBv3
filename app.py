from typing import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Any
import requests
from io import StringIO
import pandas as pd
from collections import defaultdict
from datetime import datetime
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
import sys

# ===== BEGIN utils/formatting.py =====
def stoplight(delta: float, better_high: bool = True, tol: float = 1e-9) -> str:
    if abs(delta) <= tol:
        return "🟡"
    improving = delta > 0 if better_high else delta < 0
    return "🟢" if improving else "🔴"

def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0

def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"

def round_or_none(value, digits: int = 2):
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except Exception:
        return None

def trend_vs_previous(current: float, previous: float, better_high: bool = True) -> str:
    return stoplight(current - previous, better_high=better_high)

def confidence_label(score: float) -> str:
    if score >= 0.75:
        return "High"
    if score >= 0.55:
        return "Medium"
    return "Low"

def first_valid(values: Iterable):
    for value in values:
        if value not in (None, "", float("nan")):
            return value
    return None
# ===== END utils/formatting.py =====

# ===== BEGIN models/constants.py =====
TEAM_ID_DEFAULT = 118  # Royals
SEASON_DEFAULT = 2026
LEAGUE_MAP = {
    "AL": 103,
    "NL": 104,
}
PITCH_TYPE_MAP = {
    "FF": "4-Seam",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "ST": "Sweeper",
    "CU": "Curve",
    "KC": "Knuckle Curve",
    "CH": "Changeup",
    "FS": "Splitter",
    "SV": "Slurve",
    "CS": "Slow Curve",
    "FA": "Fastball",
}
SECTION_HELP = {
    "team_summary": "Overall team effectiveness and scoring balance across the season.",
    "rolling": "Short-term trend indicator using the last three completed games.",
    "inning": "Run production and prevention by inning across completed games.",
    "players": "Weighted offensive contribution ranking with trend and consistency overlays.",
    "pitching": "Run prevention effectiveness for starters and bullpen.",
    "advanced": "Statcast-style contact quality, spin, and pitch mix metrics where available.",
    "forecast": "Estimated playoff outlook and next-five-games win probabilities.",
}
# ===== END models/constants.py =====

# ===== BEGIN api/mlb_client.py =====
BASE_URL = "https://statsapi.mlb.com/api/v1"
TIMEOUT = 30

class MLBApiError(RuntimeError):
    pass

@dataclass
class MLBClient:
    session: requests.Session | None = None

    def __post_init__(self):
        if self.session is None:
            self.session = requests.Session()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{BASE_URL}{path}"
        response = self.session.get(url, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()

    def get_teams(self) -> list[dict[str, Any]]:
        data = self._get("/teams", {"sportId": 1})
        return data.get("teams", [])

    def get_team(self, team_id: int) -> dict[str, Any] | None:
        teams = self.get_teams()
        return next((team for team in teams if team["id"] == team_id), None)

    def get_schedule(
        self,
        team_id: int,
        season: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"sportId": 1, "teamId": team_id, "season": season}
        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date
        data = self._get("/schedule", params)
        games: list[dict[str, Any]] = []
        for day in data.get("dates", []):
            games.extend(day.get("games", []))
        return games

    def get_standings(self, league_id: int, season: int) -> dict[str, Any]:
        return self._get("/standings", {"leagueId": league_id, "season": season})

    def get_wild_card(self, league_id: int, season: int) -> dict[str, Any]:
        return self._get(f"/standings/wildCard", {"leagueId": league_id, "season": season})

    def get_game_feed(self, game_pk: int) -> dict[str, Any]:
        return self._get(f"/game/{game_pk}/feed/live")

    def get_boxscore(self, game_pk: int) -> dict[str, Any]:
        return self._get(f"/game/{game_pk}/boxscore")

    def get_linescore(self, game_pk: int) -> dict[str, Any]:
        return self._get(f"/game/{game_pk}/linescore")

    def get_play_by_play(self, game_pk: int) -> dict[str, Any]:
        return self._get(f"/game/{game_pk}/playByPlay")
# ===== END api/mlb_client.py =====

# ===== BEGIN api/statcast_client.py =====
BASE_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
TIMEOUT = 60

class StatcastClient:
    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()

    def fetch_csv(self, params: dict[str, Any]) -> pd.DataFrame:
        response = self.session.get(BASE_URL, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        text = response.text.strip()
        if not text:
            return pd.DataFrame()
        return pd.read_csv(StringIO(text))

    def team_events(self, season: int, team_abbrev: str) -> pd.DataFrame:
        return self.fetch_csv(
            {
                "year": season,
                "type": "details",
                "team": team_abbrev,
            }
        )

    def hitter_events(self, season: int, player_id: int) -> pd.DataFrame:
        return self.fetch_csv(
            {
                "year": season,
                "player_type": "batter",
                "player_id": player_id,
                "type": "details",
            }
        )

    def pitcher_events(self, season: int, player_id: int) -> pd.DataFrame:
        return self.fetch_csv(
            {
                "year": season,
                "player_type": "pitcher",
                "player_id": player_id,
                "type": "details",
            }
        )
# ===== END api/statcast_client.py =====

# ===== BEGIN services/team_stats.py =====
FINAL_STATES = {"Final", "Game Over", "Completed Early"}

@dataclass
class TeamContext:
    team_id: int
    team_name: str
    team_abbrev: str
    league_id: int

def extract_team_context(team: dict[str, Any]) -> TeamContext:
    return TeamContext(
        team_id=team["id"],
        team_name=team["name"],
        team_abbrev=team.get("abbreviation", team.get("fileCode", "").upper()),
        league_id=team.get("league", {}).get("id", 103),
    )

def is_completed_game(game: dict[str, Any]) -> bool:
    return game.get("status", {}).get("detailedState") in FINAL_STATES

def normalize_schedule_row(game: dict[str, Any], team_id: int) -> dict[str, Any]:
    teams = game["teams"]
    is_home = teams["home"]["team"]["id"] == team_id
    team_side = "home" if is_home else "away"
    opp_side = "away" if is_home else "home"
    team_score = teams[team_side].get("score")
    opp_score = teams[opp_side].get("score")
    return {
        "gamePk": game["gamePk"],
        "date": game["gameDate"],
        "status": game.get("status", {}).get("detailedState"),
        "is_home": is_home,
        "opponent": teams[opp_side]["team"]["name"],
        "opponent_id": teams[opp_side]["team"]["id"],
        "team_score": team_score,
        "opp_score": opp_score,
        "won": None if team_score is None or opp_score is None else team_score > opp_score,
    }

def build_game_log(schedule: list[dict[str, Any]], team_id: int) -> pd.DataFrame:
    rows = [normalize_schedule_row(game, team_id) for game in schedule]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df

def parse_game_offense(feed: dict[str, Any], team_id: int) -> dict[str, Any]:
    box = feed.get("liveData", {}).get("boxscore", {}).get("teams", {})
    side = "home" if feed["gameData"]["teams"]["home"]["id"] == team_id else "away"
    team_box = box.get(side, {})
    batting_order = team_box.get("batters", [])
    players = team_box.get("players", {})
    rows: list[dict[str, Any]] = []
    totals = defaultdict(float)
    for pid in batting_order:
        player = players.get(f"ID{pid}", {})
        stats = player.get("stats", {}).get("batting", {})
        name = player.get("person", {}).get("fullName")
        if not name:
            continue
        row = {
            "player_id": pid,
            "player": name,
            "ab": stats.get("atBats", 0),
            "h": stats.get("hits", 0),
            "bb": stats.get("baseOnBalls", 0),
            "hbp": stats.get("hitByPitch", 0),
            "2b": stats.get("doubles", 0),
            "3b": stats.get("triples", 0),
            "hr": stats.get("homeRuns", 0),
            "rbi": stats.get("rbi", 0),
            "runs": stats.get("runs", 0),
            "so": stats.get("strikeOuts", 0),
            "tb": stats.get("totalBases", 0),
            "sb": stats.get("stolenBases", 0),
            "lob": stats.get("leftOnBase", 0),
            "game_pk": feed["gamePk"],
            "date": feed["gameData"]["datetime"]["officialDate"],
        }
        row["ob_events"] = row["h"] + row["bb"] + row["hbp"]
        row["xbh"] = row["2b"] + row["3b"] + row["hr"]
        row["productive_outs"] = 0
        rows.append(row)
        for k, v in row.items():
            if isinstance(v, (int, float)):
                totals[k] += v
    return {"player_rows": rows, "team_totals": dict(totals)}

def parse_game_pitching(feed: dict[str, Any], team_id: int) -> list[dict[str, Any]]:
    box = feed.get("liveData", {}).get("boxscore", {}).get("teams", {})
    side = "home" if feed["gameData"]["teams"]["home"]["id"] == team_id else "away"
    team_box = box.get(side, {})
    pitchers = team_box.get("pitchers", [])
    players = team_box.get("players", {})
    rows: list[dict[str, Any]] = []
    for pid in pitchers:
        player = players.get(f"ID{pid}", {})
        stats = player.get("stats", {}).get("pitching", {})
        name = player.get("person", {}).get("fullName")
        if not name:
            continue
        rows.append(
            {
                "player_id": pid,
                "pitcher": name,
                "ip": stats.get("inningsPitched", "0.0"),
                "er": stats.get("earnedRuns", 0),
                "so": stats.get("strikeOuts", 0),
                "bb": stats.get("baseOnBalls", 0),
                "hr": stats.get("homeRuns", 0),
                "hits": stats.get("hits", 0),
                "pitches": stats.get("numberOfPitches", 0),
                "strikes": stats.get("strikes", 0),
                "date": feed["gameData"]["datetime"]["officialDate"],
                "game_pk": feed["gamePk"],
            }
        )
    return rows

def parse_linescore(feed: dict[str, Any], team_id: int) -> list[dict[str, Any]]:
    innings = feed.get("liveData", {}).get("linescore", {}).get("innings", [])
    is_home = feed["gameData"]["teams"]["home"]["id"] == team_id
    rows = []
    for idx, inning in enumerate(innings, start=1):
        team_runs = inning.get("home" if is_home else "away", {}).get("runs", 0)
        opp_runs = inning.get("away" if is_home else "home", {}).get("runs", 0)
        rows.append({"inning": idx, "runs_for": team_runs, "runs_against": opp_runs, "game_pk": feed["gamePk"]})
    return rows

def parse_momentum(feed: dict[str, Any], team_id: int) -> list[dict[str, Any]]:
    plays = feed.get("liveData", {}).get("plays", {}).get("allPlays", [])
    rows = []
    home_id = feed["gameData"]["teams"]["home"]["id"]
    for play in plays:
        result = play.get("result", {})
        about = play.get("about", {})
        matchup = play.get("matchup", {})
        scoring = result.get("isScoringPlay")
        if not scoring and result.get("eventType") not in {"home_run", "triple", "double", "walk", "strikeout"}:
            continue
        batter_team_home = matchup.get("batSide") is not None and about.get("isTopInning", False) != (home_id == team_id)
        impact_level = "medium"
        if scoring:
            runs_scored = len(play.get("runners", []))
            impact_level = "very high" if runs_scored >= 2 else "high"
        rows.append(
            {
                "game_pk": feed["gamePk"],
                "inning": about.get("inning"),
                "half": "Top" if about.get("isTopInning") else "Bottom",
                "event": result.get("description", result.get("event", "")),
                "event_type": result.get("eventType"),
                "impact": impact_level,
                "team_offense": batter_team_home,
            }
        )
    return rows

def aggregate_innings(linescore_rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(linescore_rows)
    if df.empty:
        return pd.DataFrame(columns=["inning", "runs_scored", "runs_allowed", "net_impact"])
    grouped = df.groupby("inning", as_index=False).agg(runs_scored=("runs_for", "sum"), runs_allowed=("runs_against", "sum"))
    grouped["net_impact"] = grouped["runs_scored"] - grouped["runs_allowed"]
    return grouped

def summarize_team(game_log: pd.DataFrame) -> dict[str, Any]:
    completed = game_log.dropna(subset=["team_score", "opp_score"]).copy()
    wins = int(completed["won"].sum()) if not completed.empty else 0
    losses = int((~completed["won"]).sum()) if not completed.empty else 0
    runs_scored = float(completed["team_score"].sum()) if not completed.empty else 0.0
    runs_allowed = float(completed["opp_score"].sum()) if not completed.empty else 0.0
    consistency = 100 - min(100, np.std(completed["team_score"]) * 18) if len(completed) > 1 else 50
    clutch = min(100, safe_div(runs_scored, max(1.0, len(completed))) * 20)
    expected_runs = safe_div(completed["team_score"].tail(3).sum(), min(3, len(completed))) + 0.8 if not completed.empty else 0.0
    return {
        "record": f"{wins}-{losses}",
        "runs_scored": int(runs_scored),
        "runs_allowed": int(runs_allowed),
        "run_differential": int(runs_scored - runs_allowed),
        "avg_runs": round(safe_div(runs_scored, len(completed)), 2),
        "avg_runs_allowed": round(safe_div(runs_allowed, len(completed)), 2),
        "expected_runs": round(expected_runs, 2),
        "consistency_rating": round(consistency, 1),
        "clutch_index": round(clutch, 1),
    }

def rolling_3(game_log: pd.DataFrame) -> dict[str, Any]:
    completed = game_log.dropna(subset=["team_score", "opp_score"]).tail(3)
    if completed.empty:
        return {}
    return {
        "Runs": round(completed["team_score"].mean(), 2),
        "Runs allowed": round(completed["opp_score"].mean(), 2),
    }
# ===== END services/team_stats.py =====

# ===== BEGIN services/player_metrics.py =====
def build_player_game_log(player_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(player_rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df

def _impact_score(df: pd.DataFrame) -> pd.Series:
    # Weighted 0-100 score
    ob_component = df["ob_events"] * 8
    xbh_component = df["xbh"] * 12
    run_component = (df["rbi"] + df["runs"]) * 5
    so_component = np.maximum(0, 12 - df["so"] * 3)
    clutch_component = (df["rbi"] * 6) + (df["productive_outs"] * 4)
    raw = 0.35 * ob_component + 0.25 * xbh_component + 0.20 * run_component + 0.10 * so_component + 0.10 * clutch_component
    return raw.clip(0, 100)

def _consistency(group: pd.DataFrame) -> float:
    if len(group) == 1:
        return 55.0
    ob_var = group["ob_events"].std(ddof=0)
    so_var = group["so"].std(ddof=0)
    impact_var = group["impact_score"].std(ddof=0)
    zero_games = (group["impact_score"] <= 20).mean()
    score = 100 - (ob_var * 18 + so_var * 10 + impact_var * 0.45 + zero_games * 25)
    return max(0.0, min(100.0, score))

def _clutch(group: pd.DataFrame) -> float:
    rbi = group["rbi"].sum() * 8
    prod = group["productive_outs"].sum() * 7
    xbh = group["xbh"].sum() * 9
    late = group.loc[group["date"] >= group["date"].max() - pd.Timedelta(days=7), "rbi"].sum() * 4
    contact = max(0, group["h"].sum() * 3 - group["so"].sum())
    return min(100.0, 0.30 * rbi + 0.20 * prod + 0.20 * xbh + 0.15 * late + 0.15 * contact)

def summarize_players(player_game_log: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if player_game_log.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    df = player_game_log.copy()
    df["impact_score"] = _impact_score(df)
    agg = df.groupby(["player_id", "player"], as_index=False).agg(
        ob_events=("ob_events", "sum"),
        xbh=("xbh", "sum"),
        so=("so", "sum"),
        rbi=("rbi", "sum"),
        runs=("runs", "sum"),
        impact=("impact_score", "mean"),
    )
    consistency_rows = []
    clutch_rows = []
    trend_rows = []
    for (player_id, player), group in df.groupby(["player_id", "player"]):
        consistency = _consistency(group)
        clutch = _clutch(group)
        last_two = group.sort_values("date").tail(2)["impact_score"].mean()
        prev = group.sort_values("date").head(max(1, len(group) - 2))["impact_score"].mean() if len(group) > 2 else group["impact_score"].iloc[0]
        trend = stoplight(last_two - prev)
        consistency_rows.append({"player_id": player_id, "player": player, "consistency_score": round(consistency, 1), "consistency_trend": trend})
        clutch_rows.append({"player_id": player_id, "player": player, "clutch_score": round(clutch, 1), "clutch_trend": trend})
        trend_rows.append({"player_id": player_id, "player": player, "trend": trend})
    out = agg.merge(pd.DataFrame(consistency_rows), on=["player_id", "player"]).merge(pd.DataFrame(clutch_rows), on=["player_id", "player"]).merge(pd.DataFrame(trend_rows), on=["player_id", "player"])
    out["impact"] = out["impact"].round(1)
    out["grade"] = pd.cut(out["impact"], bins=[-1, 29.99, 49.99, 64.99, 79.99, 100], labels=["F", "D", "C", "B", "A"])
    out = out.sort_values(["impact", "consistency_score"], ascending=False).reset_index(drop=True)
    return out, pd.DataFrame(clutch_rows).sort_values("clutch_score", ascending=False), pd.DataFrame(consistency_rows).sort_values("consistency_score", ascending=False)
# ===== END services/player_metrics.py =====

# ===== BEGIN services/pitching_metrics.py =====
def innings_to_float(ip: str | float | int) -> float:
    if isinstance(ip, (float, int)):
        return float(ip)
    whole, _, frac = str(ip).partition(".")
    outs = int(whole) * 3 + int(frac or 0)
    return outs / 3

def summarize_pitchers(pitch_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(pitch_rows)
    if df.empty:
        return df
    df["ip_float"] = df["ip"].map(innings_to_float)
    grouped = df.groupby(["player_id", "pitcher"], as_index=False).agg(
        ip=("ip_float", "sum"),
        er=("er", "sum"),
        so=("so", "sum"),
        bb=("bb", "sum"),
        hr=("hr", "sum"),
        hits=("hits", "sum"),
    )
    grouped["impact"] = (
        100
        - grouped["er"] * 12
        - grouped["bb"] * 4
        - grouped["hr"] * 8
        + grouped["so"] * 3
        + grouped["ip"] * 2
    ).clip(0, 100)
    trend_rows = []
    for (player_id, pitcher), group in df.groupby(["player_id", "pitcher"]):
        recent = group.tail(1)
        earlier = group.head(max(1, len(group) - 1))
        recent_score = (100 - recent["er"].sum() * 12 - recent["bb"].sum() * 4 - recent["hr"].sum() * 8 + recent["so"].sum() * 3)
        earlier_score = (100 - earlier["er"].sum() * 12 - earlier["bb"].sum() * 4 - earlier["hr"].sum() * 8 + earlier["so"].sum() * 3)
        trend_rows.append({"player_id": player_id, "pitcher": pitcher, "trend": stoplight(recent_score - earlier_score)})
    grouped = grouped.merge(pd.DataFrame(trend_rows), on=["player_id", "pitcher"], how="left")
    grouped["ip"] = grouped["ip"].round(1)
    grouped["impact"] = grouped["impact"].round(1)
    return grouped.sort_values("impact", ascending=False).reset_index(drop=True)
# ===== END services/pitching_metrics.py =====

# ===== BEGIN services/advanced_metrics.py =====
def _numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def summarize_hitter_statcast(events: pd.DataFrame, player_lookup: dict[int, str], team_abbrev: str) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["player_id", "player", "avg_exit_velocity", "hard_hit_pct", "avg_hr_distance", "home_hr_dist", "away_hr_dist", "trend"])
    df = events.copy()
    df = _numeric(df, ["launch_speed", "hit_distance_sc"])
    if "batter" not in df.columns:
        return pd.DataFrame()
    rows = []
    for pid, group in df.groupby("batter"):
        player = player_lookup.get(int(pid), str(pid))
        batted = group.dropna(subset=["launch_speed"])
        hr = group[group.get("events", "").eq("home_run") if "events" in group.columns else False]
        home_hr = hr[hr.get("home_team") == team_abbrev] if "home_team" in hr.columns else pd.DataFrame()
        away_hr = hr[hr.get("away_team") == team_abbrev] if "away_team" in hr.columns else pd.DataFrame()
        recent = batted.tail(max(1, len(batted) // 3))["launch_speed"].mean() if not batted.empty else np.nan
        earlier = batted.head(max(1, len(batted) - max(1, len(batted) // 3)))["launch_speed"].mean() if not batted.empty else np.nan
        rows.append(
            {
                "player_id": int(pid),
                "player": player,
                "avg_exit_velocity": round(float(batted["launch_speed"].mean()), 1) if not batted.empty else None,
                "hard_hit_pct": round(float((batted["launch_speed"] >= 95).mean() * 100), 1) if not batted.empty else None,
                "avg_hr_distance": round(float(hr["hit_distance_sc"].mean()), 1) if not hr.empty and "hit_distance_sc" in hr.columns else None,
                "home_hr_dist": round(float(home_hr["hit_distance_sc"].mean()), 1) if not home_hr.empty and "hit_distance_sc" in home_hr.columns else None,
                "away_hr_dist": round(float(away_hr["hit_distance_sc"].mean()), 1) if not away_hr.empty and "hit_distance_sc" in away_hr.columns else None,
                "trend": stoplight((recent or 0) - (earlier or 0)),
            }
        )
    return pd.DataFrame(rows).sort_values("avg_exit_velocity", ascending=False, na_position="last")

def summarize_pitcher_statcast(events: pd.DataFrame, player_lookup: dict[int, str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if events.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    df = events.copy()
    df = _numeric(df, ["release_spin_rate", "release_speed", "launch_speed"])
    player_col = "pitcher" if "pitcher" in df.columns else None
    if player_col is None:
        return pd.DataFrame(), pd.DataFrame(), {}
    pitcher_rows = []
    pitch_mix_rows = []
    for pid, group in df.groupby(player_col):
        player = player_lookup.get(int(pid), str(pid))
        recent = group.tail(max(1, len(group) // 3))["release_spin_rate"].mean()
        earlier = group.head(max(1, len(group) - max(1, len(group) // 3)))["release_spin_rate"].mean()
        pitcher_rows.append(
            {
                "player_id": int(pid),
                "pitcher": player,
                "avg_exit_velocity_allowed": round(float(group["launch_speed"].mean()), 1) if "launch_speed" in group else None,
                "avg_spin_rate": round(float(group["release_spin_rate"].mean()), 1) if "release_spin_rate" in group else None,
                "avg_velocity": round(float(group["release_speed"].mean()), 1) if "release_speed" in group else None,
                "trend": stoplight((recent or 0) - (earlier or 0)),
            }
        )
        if "pitch_type" in group.columns:
            counts = group["pitch_type"].fillna("UNK").value_counts(normalize=True)
            for pitch_type, pct in counts.items():
                sub = group[group["pitch_type"] == pitch_type]
                pitch_mix_rows.append(
                    {
                        "player_id": int(pid),
                        "pitcher": player,
                        "pitch_type": PITCH_TYPE_MAP.get(pitch_type, pitch_type),
                        "usage_pct": round(pct * 100, 1),
                        "avg_spin_rate": round(float(sub["release_spin_rate"].mean()), 1) if "release_spin_rate" in sub else None,
                        "avg_velocity": round(float(sub["release_speed"].mean()), 1) if "release_speed" in sub else None,
                    }
                )
    team_summary = {
        "team_avg_exit_velocity": round(float(df["launch_speed"].mean()), 1) if "launch_speed" in df.columns else None,
        "team_avg_spin_rate": round(float(df["release_spin_rate"].mean()), 1) if "release_spin_rate" in df.columns else None,
    }
    return pd.DataFrame(pitcher_rows).sort_values("avg_spin_rate", ascending=False, na_position="last"), pd.DataFrame(pitch_mix_rows), team_summary
# ===== END services/advanced_metrics.py =====

# ===== BEGIN services/inning_analysis.py =====
def heatmap_labels(values: pd.Series, against: bool = False) -> list[str]:
    labels = []
    for val in values:
        if val == 0:
            labels.append("⬜")
        elif val == 1:
            labels.append("🟦")
        elif val == 2:
            labels.append("🟩")
        else:
            labels.append("🟥")
    return labels

def build_heatmap_table(inning_df: pd.DataFrame, col: str) -> pd.DataFrame:
    if inning_df.empty:
        return pd.DataFrame()
    row = inning_df.set_index("inning")[col]
    intensity = heatmap_labels(row)
    return pd.DataFrame([row.tolist(), intensity], index=["Runs", "Intensity"], columns=row.index.tolist())
# ===== END services/inning_analysis.py =====

# ===== BEGIN services/momentum.py =====
def summarize_momentum(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df[["game_pk", "inning", "event", "impact"]].rename(columns={"game_pk": "game"}).head(20)
# ===== END services/momentum.py =====

# ===== BEGIN services/lineup_model.py =====
def recommend_lineup(player_summary: pd.DataFrame) -> pd.DataFrame:
    if player_summary.empty:
        return pd.DataFrame()
    df = player_summary.copy()
    df["lineup_score"] = (
        df["ob_events"] * 2.0
        + df["impact"] * 0.8
        + df["xbh"] * 3.0
        - df["so"] * 0.6
        + df["clutch_score"] * 0.2
    )
    df = df.sort_values("lineup_score", ascending=False).reset_index(drop=True)
    df["slot"] = range(1, len(df) + 1)
    max_score = max(df["lineup_score"].max(), 1)
    df["confidence"] = df["lineup_score"].map(lambda x: confidence_label(x / max_score))
    return df[["slot", "player", "confidence"]]
# ===== END services/lineup_model.py =====

# ===== BEGIN services/playoff_model.py =====
def estimate_playoff_probability(team_summary: dict, standings_row: dict | None, wild_row: dict | None) -> dict:
    wins, losses = [int(x) for x in team_summary["record"].split("-")]
    games = max(1, wins + losses)
    win_pct = wins / games
    run_diff = team_summary["run_differential"]
    score = 0.45 * win_pct + 0.20 * max(-0.2, min(0.2, run_diff / 100)) + 0.20 * min(1.0, team_summary["consistency_rating"] / 100) + 0.15 * min(1.0, team_summary["clutch_index"] / 100)
    division_rank = standings_row.get("divisionRank") if standings_row else None
    wc_rank = wild_row.get("wildCardRank") if wild_row else None
    if division_rank:
        score += max(0, (6 - int(division_rank))) * 0.02
    if wc_rank:
        score += max(0, (6 - int(wc_rank))) * 0.015
    playoff = max(0, min(100, round(score * 100, 1)))
    division = max(0, min(100, round(playoff * 0.45, 1)))
    wild = max(0, min(100, round(playoff * 0.75, 1)))
    return {
        "playoff_probability": playoff,
        "division_probability": division,
        "wild_card_probability": wild,
    }

def extract_standings_rows(standings: dict, team_id: int) -> dict | None:
    for record_set in standings.get("records", []):
        for row in record_set.get("teamRecords", []):
            if row.get("team", {}).get("id") == team_id:
                return row
    return None
# ===== END services/playoff_model.py =====

# ===== BEGIN services/game_forecast.py =====
def forecast_next_five(game_log: pd.DataFrame, upcoming: pd.DataFrame, team_summary: dict) -> pd.DataFrame:
    if upcoming.empty:
        return pd.DataFrame()
    recent = game_log.dropna(subset=["team_score", "opp_score"]).tail(10)
    avg_scored = recent["team_score"].mean() if not recent.empty else team_summary["avg_runs"]
    avg_allowed = recent["opp_score"].mean() if not recent.empty else team_summary["avg_runs_allowed"]
    home_boost = 0.04
    rows = []
    for _, row in upcoming.head(5).iterrows():
        base = 0.50
        base += (avg_scored - avg_allowed) * 0.03
        base += 0.02 if row["is_home"] else -home_boost
        win_prob = max(0.15, min(0.85, base))
        rows.append(
            {
                "date": row["date"].date().isoformat(),
                "opponent": row["opponent"],
                "home_away": "Home" if row["is_home"] else "Away",
                "win_probability": round(win_prob * 100, 1),
                "confidence": confidence_label(abs(win_prob - 0.5) * 2),
                "key_factors": "recent scoring trend, run prevention, home/away split",
            }
        )
    return pd.DataFrame(rows)
# ===== END services/game_forecast.py =====

# ===== BEGIN components/charts.py =====
PLOT_TEMPLATE = "plotly_white"

def _base_layout(fig: go.Figure, height: int = 330) -> go.Figure:
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        margin=dict(l=12, r=12, t=52, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.98)",
        font=dict(size=13),
        title=dict(x=0.02, xanchor="left"),
    )
    return fig

def playoff_chart(probabilities: dict):
    df = pd.DataFrame(
        {
            "Category": ["Playoff", "Division", "Wild Card"],
            "Probability": [
                probabilities.get("playoff_probability", 0),
                probabilities.get("division_probability", 0),
                probabilities.get("wild_card_probability", 0),
            ],
        }
    )
    fig = px.bar(df, x="Category", y="Probability", text="Probability", title="Estimated Postseason Outlook")
    fig.update_yaxes(range=[0, 100], title="Probability %")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    return _base_layout(fig)

def exit_velocity_chart(df: pd.DataFrame):
    if df.empty or "avg_exit_velocity" not in df.columns:
        return None
    plot_df = (
        df.dropna(subset=["avg_exit_velocity"])
        .sort_values("avg_exit_velocity", ascending=False)
        .head(12)
    )
    if plot_df.empty:
        return None
    fig = px.bar(
        plot_df,
        x="avg_exit_velocity",
        y="player",
        orientation="h",
        title="Average Exit Velocity Leaders",
        text="avg_exit_velocity",
    )
    fig.update_traces(texttemplate="%{text:.1f} mph", textposition="outside")
    fig.update_yaxes(categoryorder="total ascending")
    fig.update_xaxes(title="Average Exit Velocity")
    return _base_layout(fig, height=420)

def pitch_mix_chart(df: pd.DataFrame, pitcher: str):
    plot_df = df[df["pitcher"] == pitcher].copy()
    if plot_df.empty:
        return None
    fig = px.pie(plot_df, names="pitch_type", values="usage_pct", hole=0.5, title=f"Pitch Mix: {pitcher}")
    fig.update_traces(texttemplate="%{label}<br>%{value:.1f}%")
    return _base_layout(fig, height=380)

def win_probability_chart(df: pd.DataFrame):
    if df.empty or "win_probability" not in df.columns:
        return None
    plot_df = df.copy()
    label_col = "opponent" if "opponent" in plot_df.columns else plot_df.columns[0]
    fig = px.bar(
        plot_df,
        x=label_col,
        y="win_probability",
        title="Next 5 Games Win Likelihood",
        text="win_probability",
    )
    fig.update_yaxes(title="Win Probability %", range=[0, 100])
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    return _base_layout(fig)

def rolling_runs_chart(game_log: pd.DataFrame):
    if game_log.empty or "team_score" not in game_log.columns:
        return None
    plot_df = game_log.dropna(subset=["team_score", "opp_score"]).copy().tail(15)
    if plot_df.empty:
        return None
    if "game_date" in plot_df.columns:
        plot_df["label"] = pd.to_datetime(plot_df["game_date"]).dt.strftime("%m/%d")
    else:
        plot_df["label"] = range(1, len(plot_df) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df["label"], y=plot_df["team_score"], mode="lines+markers", name="Runs For"))
    fig.add_trace(go.Scatter(x=plot_df["label"], y=plot_df["opp_score"], mode="lines+markers", name="Runs Against"))
    fig.update_yaxes(title="Runs")
    fig.update_xaxes(title="Game")
    fig.update_layout(title="Recent Game Trend")
    return _base_layout(fig, height=360)
# ===== END components/charts.py =====

# ===== BEGIN components/heatmaps.py =====
def inning_heatmap(inning_df: pd.DataFrame, value_col: str, title: str):
    if inning_df.empty or value_col not in inning_df.columns:
        return None
    z = [inning_df[value_col].tolist()]
    x = inning_df["inning"].tolist()
    custom = [[f"Inning {i}: {v}" for i, v in zip(x, inning_df[value_col].tolist())]]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=[title],
            text=custom,
            texttemplate="%{z}",
            textfont={"size": 14},
            hovertemplate="%{text}<extra></extra>",
            colorscale="Blues",
            showscale=False,
            xgap=4,
            ygap=4,
        )
    )
    fig.update_layout(
        title=title,
        height=220,
        margin=dict(l=10, r=10, t=48, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.98)",
    )
    fig.update_xaxes(title=None, side="top")
    fig.update_yaxes(title=None)
    return fig
# ===== END components/heatmaps.py =====

# ===== BEGIN components/tables.py =====
STOPLIGHT_STYLE = {
    "🟢": "background-color: rgba(34, 197, 94, 0.18); color: #166534; font-weight: 700; text-align: center;",
    "🟡": "background-color: rgba(245, 158, 11, 0.20); color: #92400e; font-weight: 700; text-align: center;",
    "🔴": "background-color: rgba(239, 68, 68, 0.18); color: #991b1b; font-weight: 700; text-align: center;",
}

def _style_stoplights(df: pd.DataFrame):
    stoplight_cols = [
        col
        for col in df.columns
        if "trend" in str(col).lower() or str(col).lower() in {"signal", "status"}
    ]
    if not stoplight_cols:
        return df

    styled = df.style
    for col in stoplight_cols:
        styled = styled.map(lambda v: STOPLIGHT_STYLE.get(v, ""), subset=[col])
    return styled

def show_table(df: pd.DataFrame, use_container_width: bool = True, height: int | None = None):
    if df is None or df.empty:
        st.info("No data available for this section.")
        return

    display_df = df.copy()
    styled = _style_stoplights(display_df)
    st.dataframe(styled, use_container_width=use_container_width, hide_index=True, height=height)
# ===== END components/tables.py =====

# ===== BEGIN app.py =====
from datetime import date

import pandas as pd
import streamlit as st

st.set_page_config(page_title="MLB Dashboard", page_icon="⚾", layout="wide")

CSS = """
<style>
    .stApp {
        background: linear-gradient(180deg, #f7f9fc 0%, #eef3f9 100%);
    }
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }
    .hero {
        background: linear-gradient(135deg, #ffffff 0%, #f4f8ff 100%);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 20px;
        padding: 20px 24px;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
    }
    .hero p {
        margin: 0.35rem 0 0 0;
        color: #475569;
    }
    .section-card {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 18px;
        padding: 14px 16px 16px 16px;
        box-shadow: 0 6px 22px rgba(15, 23, 42, 0.05);
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 1.08rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
        color: #0f172a;
    }
    .section-subtitle {
        color: #475569;
        font-size: 0.94rem;
        margin-bottom: 0.75rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.96);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 18px;
        padding: 0.65rem 0.85rem;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
    }
    div[data-testid="stMetricLabel"] {
        color: #64748b;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #0f172a;
    }
    .sidebar-note {
        background: rgba(255,255,255,0.9);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 14px;
        padding: 12px;
        font-size: 0.92rem;
        color: #475569;
        margin-top: 0.75rem;
    }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <h1>Live MLB Analytics Dashboard</h1>
        <p>Streamlit app powered by MLB StatsAPI and Baseball Savant Statcast data with a cleaner executive-style UI.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

mlb = MLBClient()
statcast = StatcastClient()

@st.cache_data(ttl=3600, show_spinner=False)
def load_teams():
    teams = mlb.get_teams()
    df = pd.DataFrame(
        [
            {
                "id": t["id"],
                "name": t["name"],
                "abbrev": t.get("abbreviation", t.get("fileCode", "").upper()),
                "league_id": t.get("league", {}).get("id", 103),
            }
            for t in teams
        ]
    ).sort_values("name")
    return df

@st.cache_data(ttl=900, show_spinner=True)
def load_dashboard(team_id: int, season: int, start_date: str | None, end_date: str | None):
    schedule = mlb.get_schedule(team_id, season, start_date, end_date)
    team = mlb.get_team(team_id)
    if team is None:
        raise ValueError("Team not found")
    context = extract_team_context(team)
    game_log = build_game_log(schedule, team_id)
    completed_games = [g for g in schedule if is_completed_game(g)]

    player_rows = []
    pitch_rows = []
    inning_rows = []
    momentum_rows = []
    player_lookup = {}
    for game in completed_games:
        feed = mlb.get_game_feed(game["gamePk"])
        offense = parse_game_offense(feed, team_id)
        player_rows.extend(offense["player_rows"])
        for row in offense["player_rows"]:
            player_lookup[row["player_id"]] = row["player"]
        pitch_rows.extend(parse_game_pitching(feed, team_id))
        inning_rows.extend(parse_linescore(feed, team_id))
        momentum_rows.extend(parse_momentum(feed, team_id))

    player_game_log = build_player_game_log(player_rows)
    player_summary, clutch_df, consistency_df = summarize_players(player_game_log)
    pitching_df = summarize_pitchers(pitch_rows)
    inning_df = aggregate_innings(inning_rows)
    team_summary = summarize_team(game_log)

    standings = mlb.get_standings(context.league_id, season)
    wild = mlb.get_wild_card(context.league_id, season)
    standings_row = extract_standings_rows(standings, team_id)
    wild_row = extract_standings_rows(wild, team_id)
    playoff_probs = estimate_playoff_probability(team_summary, standings_row, wild_row)

    upcoming = game_log[game_log["team_score"].isna()].copy()
    next_five = forecast_next_five(game_log, upcoming, team_summary)

    hitter_statcast = pd.DataFrame()
    pitcher_statcast = pd.DataFrame()
    pitch_mix = pd.DataFrame()
    team_statcast_summary = {}
    try:
        team_events = statcast.team_events(season, context.team_abbrev)
        hitter_statcast = summarize_hitter_statcast(team_events, player_lookup, context.team_abbrev)
        pitcher_statcast, pitch_mix, team_statcast_summary = summarize_pitcher_statcast(team_events, player_lookup)
    except Exception:
        pass

    return {
        "context": context,
        "game_log": game_log,
        "team_summary": team_summary,
        "player_summary": player_summary,
        "clutch_df": clutch_df,
        "consistency_df": consistency_df,
        "pitching_df": pitching_df,
        "inning_df": inning_df,
        "momentum_df": summarize_momentum(momentum_rows),
        "playoff_probs": playoff_probs,
        "next_five": next_five,
        "lineup_df": recommend_lineup(player_summary),
        "hitter_statcast": hitter_statcast,
        "pitcher_statcast": pitcher_statcast,
        "pitch_mix": pitch_mix,
        "team_statcast_summary": team_statcast_summary,
    }

def card_header(title: str, subtitle: str):
    st.markdown(
        f"<div class='section-card'><div class='section-title'>{title}</div><div class='section-subtitle'>{subtitle}</div>",
        unsafe_allow_html=True,
    )

def card_close():
    st.markdown("</div>", unsafe_allow_html=True)

teams_df = load_teams()
team_options = teams_df.set_index("name")["id"].to_dict()
default_team_name = teams_df.loc[teams_df["id"] == TEAM_ID_DEFAULT, "name"].iloc[0]

with st.sidebar:
    st.header("Dashboard Controls")
    team_name = st.selectbox("Team", list(team_options.keys()), index=list(team_options.keys()).index(default_team_name))
    season = st.number_input("Season", min_value=2015, max_value=2100, value=SEASON_DEFAULT, step=1)
    start_date = st.date_input("Start date", value=date(int(season), 1, 1))
    end_date = st.date_input("End date", value=date.today())
    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
    st.markdown(
        "<div class='sidebar-note'><strong>Trend colors</strong><br>🟢 improving<br>🟡 stable<br>🔴 declining</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='sidebar-note'><strong>Data note</strong><br>Statcast-driven sections gracefully show unavailable data when source coverage is incomplete.</div>",
        unsafe_allow_html=True,
    )

team_id = int(team_options[team_name])
data = load_dashboard(team_id, int(season), start_date.isoformat(), end_date.isoformat())
ts = data["team_summary"]
completed_games = data["game_log"].dropna(subset=["team_score", "opp_score"])
prev_avg_runs = completed_games.head(max(1, len(completed_games) - 1))["team_score"].mean() if not completed_games.empty else 0
prev_avg_allowed = completed_games.head(max(1, len(completed_games) - 1))["opp_score"].mean() if not completed_games.empty else 0

metric_cols = st.columns(6)
metric_cols[0].metric("Record", ts.get("record", "0-0"))
metric_cols[1].metric("Runs Scored", ts.get("runs_scored", 0))
metric_cols[2].metric("Runs Allowed", ts.get("runs_allowed", 0))
metric_cols[3].metric("Run Diff", ts.get("run_differential", 0))
metric_cols[4].metric("Expected Runs", ts.get("expected_runs", 0))
metric_cols[5].metric("Consistency", ts.get("consistency_rating", 0))

team_tab, trend_tab, player_tab, pitch_tab, inning_tab, momentum_tab, lineup_tab, advanced_tab, forecast_tab = st.tabs(
    [
        "Team Summary",
        "Trends",
        "Player Performance",
        "Pitching",
        "Inning Analysis",
        "Momentum",
        "Lineup",
        "Advanced Metrics",
        "Forecast",
    ]
)

with team_tab:
    card_header("Team Performance Summary", SECTION_HELP["team_summary"])
    summary_df = pd.DataFrame(
        [
            {"Metric": "Record", "Value": ts["record"], "Trend": "🟢" if ts["run_differential"] >= 0 else "🟡"},
            {"Metric": "Runs scored", "Value": ts["runs_scored"], "Trend": stoplight(ts["avg_runs"] - prev_avg_runs)},
            {"Metric": "Runs allowed", "Value": ts["runs_allowed"], "Trend": stoplight(ts["avg_runs_allowed"] - prev_avg_allowed, better_high=False)},
            {"Metric": "Run differential", "Value": ts["run_differential"], "Trend": stoplight(ts["run_differential"])},
            {"Metric": "Avg runs per game", "Value": ts["avg_runs"], "Trend": stoplight(ts["avg_runs"] - prev_avg_runs)},
            {"Metric": "Avg runs allowed", "Value": ts["avg_runs_allowed"], "Trend": stoplight(ts["avg_runs_allowed"] - prev_avg_allowed, better_high=False)},
            {"Metric": "Expected runs next game", "Value": ts["expected_runs"], "Trend": stoplight(ts["expected_runs"] - ts["avg_runs"])},
            {"Metric": "Consistency rating", "Value": ts["consistency_rating"], "Trend": stoplight(ts["consistency_rating"] - 50)},
            {"Metric": "Clutch index", "Value": ts["clutch_index"], "Trend": stoplight(ts["clutch_index"] - 45)},
        ]
    )
    show_table(summary_df, height=390)
    card_close()

with trend_tab:
    c1, c2 = st.columns([1.1, 1.4])
    with c1:
        card_header("Rolling 3 Game Average", SECTION_HELP["rolling"])
        completed = completed_games.tail(3)
        rolling_df = pd.DataFrame(
            [
                {"Metric": "Runs", "Value": round(completed["team_score"].mean(), 2) if not completed.empty else 0, "Trend": stoplight((completed["team_score"].mean() if not completed.empty else 0) - ts["avg_runs"])},
                {"Metric": "Runs allowed", "Value": round(completed["opp_score"].mean(), 2) if not completed.empty else 0, "Trend": stoplight((completed["opp_score"].mean() if not completed.empty else 0) - ts["avg_runs_allowed"], better_high=False)},
            ]
        )
        show_table(rolling_df, height=180)
        card_close()
    with c2:
        card_header("Recent Game Trend", "Last completed games with runs for and against.")
        fig = rolling_runs_chart(data["game_log"])
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough completed games to render trend chart.")
        card_close()

with player_tab:
    card_header("Player Performance Index", SECTION_HELP["players"])
    st.markdown("**Impact formula:** 35% on-base events, 25% extra-base hits, 20% run creation, 10% strikeout avoidance, 10% clutch contribution.")
    show_table(data["player_summary"][["player", "grade", "impact", "trend", "ob_events", "xbh", "so", "consistency_score"]], height=460)
    card_close()

    c1, c2 = st.columns(2)
    with c1:
        card_header("Clutch Index", "Performance in high-leverage situations.")
        show_table(data["clutch_df"][["player", "clutch_score", "clutch_trend"]], height=360)
        card_close()
    with c2:
        card_header("Consistency Rating", "Inverse volatility score using OB variation, strikeout variation, zero-production frequency, and impact fluctuation.")
        show_table(data["consistency_df"][["player", "consistency_score", "consistency_trend"]], height=360)
        card_close()

with pitch_tab:
    card_header("Pitching Performance", SECTION_HELP["pitching"])
    show_table(data["pitching_df"][["pitcher", "ip", "er", "so", "bb", "hr", "impact", "trend"]], height=420)
    card_close()

with inning_tab:
    card_header("Inning Performance Matrix", SECTION_HELP["inning"])
    show_table(data["inning_df"], height=320)
    card_close()
    c1, c2 = st.columns(2)
    with c1:
        card_header("Inning Heat Map: Runs For", "Visual intensity of scoring by inning.")
        fig = inning_heatmap(data["inning_df"], "runs_scored", "Runs For")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        show_table(build_heatmap_table(data["inning_df"], "runs_scored"), height=180)
        card_close()
    with c2:
        card_header("Inning Heat Map: Runs Against", "Visual intensity of opponent scoring by inning.")
        fig = inning_heatmap(data["inning_df"], "runs_allowed", "Runs Against")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        show_table(build_heatmap_table(data["inning_df"], "runs_allowed"), height=180)
        card_close()

with momentum_tab:
    card_header("Momentum Events", "Highest leverage moments impacting game outcome.")
    show_table(data["momentum_df"], height=420)
    card_close()

with lineup_tab:
    card_header("Lineup Optimization Model", "Ideal batting order based on on-base frequency, impact, strikeout avoidance, and extra-base production.")
    show_table(data["lineup_df"], height=420)
    card_close()

with advanced_tab:
    c1, c2 = st.columns([1.2, 1])
    with c1:
        card_header("Hitter Advanced Metrics", SECTION_HELP["advanced"])
        if data["hitter_statcast"].empty:
            st.info("Hitter Statcast data unavailable for current filters.")
        else:
            show_table(data["hitter_statcast"], height=420)
            ev_fig = exit_velocity_chart(data["hitter_statcast"])
            if ev_fig:
                st.plotly_chart(ev_fig, use_container_width=True)
        card_close()
    with c2:
        card_header("Pitcher Statcast", "Exit velocity allowed, spin, velocity, and pitch mix where available.")
        if data["pitcher_statcast"].empty:
            st.info("Pitcher Statcast data unavailable for current filters.")
        else:
            show_table(data["pitcher_statcast"], height=250)
            if not data["pitch_mix"].empty:
                pitcher_name = st.selectbox(
                    "Pitch mix chart pitcher",
                    data["pitch_mix"]["pitcher"].drop_duplicates().tolist(),
                    key="pitch_mix_pitcher",
                )
                fig = pitch_mix_chart(data["pitch_mix"], pitcher_name)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                show_table(data["pitch_mix"][data["pitch_mix"]["pitcher"] == pitcher_name], height=220)
        if data["team_statcast_summary"]:
            team_stat_df = pd.DataFrame(
                [
                    {"Metric": "Team Avg EV Allowed", "Value": data["team_statcast_summary"].get("team_avg_exit_velocity")},
                    {"Metric": "Team Avg Spin Rate", "Value": data["team_statcast_summary"].get("team_avg_spin_rate")},
                ]
            )
            show_table(team_stat_df, height=120)
        card_close()

with forecast_tab:
    c1, c2 = st.columns([1, 1.2])
    with c1:
        card_header("Forecasting / Playoff Outlook", SECTION_HELP["forecast"])
        pcols = st.columns(3)
        pcols[0].metric("Playoff %", data["playoff_probs"]["playoff_probability"])
        pcols[1].metric("Division %", data["playoff_probs"]["division_probability"])
        pcols[2].metric("Wild Card %", data["playoff_probs"]["wild_card_probability"])
        st.plotly_chart(playoff_chart(data["playoff_probs"]), use_container_width=True)
        card_close()
    with c2:
        card_header("Next 5 Games Likelihood to Win", "Projected win probability for each upcoming game.")
        if data["next_five"].empty:
            st.info("No upcoming scheduled games found in current date window.")
        else:
            show_table(data["next_five"], height=280)
            win_fig = win_probability_chart(data["next_five"])
            if win_fig:
                st.plotly_chart(win_fig, use_container_width=True)
        card_close()
# ===== END app.py =====
