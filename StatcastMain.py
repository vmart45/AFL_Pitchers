import datetime
import requests
import polars as pl
from typing import Any, Dict, Iterable, List, Optional, Union

SPORT_ID = 17  # Arizona Fall League

# --------------------------- helpers ---------------------------

STRIP_TOKENS = {"details", "pitchData", "hitData", "breaks", "coordinates"}

def normalize_key(key: str, sep: str = "__") -> str:
    """
    Remove namespace tokens (details, pitchData, hitData, breaks, coordinates)
    from ANYWHERE in the flattened path. Preserve other structure like call__code.
    """
    parts = [p for p in key.split(sep) if p not in STRIP_TOKENS and p != ""]
    return sep.join(parts) if parts else key

def flatten_dict(d: Dict[str, Any], prefix: str = "", sep: str = "__") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, key, sep))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    out.update(flatten_dict(item, f"{key}{sep}{i}", sep))
                else:
                    out[normalize_key(f"{key}{sep}{i}", sep)] = item
        else:
            out[normalize_key(key, sep)] = v
    return out

def get_afl_games(date_str: str) -> List[int]:
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId={SPORT_ID}&date={date_str}"
    j = requests.get(url, timeout=30).json()
    return [g["gamePk"] for d in j.get("dates", []) for g in d.get("games", [])]

def safe_get(d: Dict[str, Any], path: Iterable[Union[str, int]], default=None):
    cur: Any = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur

# --------------------------- core scrape ---------------------------

def get_game_feed(game_pk: int) -> Optional[Dict[str, Any]]:
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    try:
        return requests.get(url, timeout=30).json()
    except Exception as e:
        print(f"⚠️ Failed to fetch game {game_pk}: {e}")
        return None

def rows_from_game(game_pk: int) -> List[Dict[str, Any]]:
    data = get_game_feed(game_pk)
    if not data:
        return []

    game_date = safe_get(data, ["gameData", "datetime", "officialDate"])
    venue_name = safe_get(data, ["gameData", "venue", "name"])
    home_name = safe_get(data, ["gameData", "teams", "home", "name"])
    away_name = safe_get(data, ["gameData", "teams", "away", "name"])

    plays = safe_get(data, ["liveData", "plays", "allPlays"], [])
    if not plays:
        print(f"⚠️ No play data for game {game_pk}")
        return []

    rows: List[Dict[str, Any]] = []

    for play_idx, play in enumerate(plays):
        about = play.get("about", {})
        matchup = play.get("matchup", {})
        count = play.get("count", {})

        base_ctx = {
            "game_id": game_pk,
            "game_date": game_date,
            "venue_name": venue_name,
            "home_team": home_name,
            "away_team": away_name,
            "at_bat_index": play.get("atBatIndex"),
            "play_idx": play_idx,
            "inning": about.get("inning"),
            "is_top_inning": about.get("isTopInning"),
            "batter_id": safe_get(matchup, ["batter", "id"]),
            "batter_name": safe_get(matchup, ["batter", "fullName"]),
            "pitcher_id": safe_get(matchup, ["pitcher", "id"]),
            "pitcher_name": safe_get(matchup, ["pitcher", "fullName"]),
            "balls": count.get("balls"),
            "strikes": count.get("strikes"),
            "outs": count.get("outs"),
        }

        for ev_idx, ev in enumerate(play.get("playEvents", [])):
            if not ev.get("isPitch"):
                continue
            row = dict(base_ctx)
            row["event_idx"] = ev_idx
            row["pitch_uid"] = f"{game_pk}-{play.get('atBatIndex')}-{ev_idx}"

            # flatten and strip namespaces (details/pitchData/hitData/breaks/coordinates)
            row.update(flatten_dict(ev.get("details") or {}))
            row.update(flatten_dict(ev.get("pitchData") or {}))
            row.update(flatten_dict(ev.get("hitData") or {}))
            rows.append(row)

    return rows

def get_afl_data(date_str: Optional[str] = None) -> pl.DataFrame:
    if date_str is None:
        date_str = datetime.date.today().strftime("%Y-%m-%d")

    game_pks = get_afl_games(date_str)
    if not game_pks:
        print(f"No AFL games for {date_str}.")
        return pl.DataFrame()

    print(f"Found {len(game_pks)} AFL games for {date_str}: {game_pks}")
    all_rows: List[Dict[str, Any]] = []
    for pk in game_pks:
        rows = rows_from_game(pk)
        if rows:
            all_rows.extend(rows)
        else:
            print(f"⚠️ No pitches for game {pk}")

    if not all_rows:
        print("⚠️ No valid pitch data.")
        return pl.DataFrame()

    df = pl.DataFrame(all_rows)
    sort_cols = [c for c in ["game_date", "game_id", "at_bat_index", "event_idx"] if c in df.columns]
    if sort_cols:
        df = df.sort(sort_cols)
    return df

# --------------------------- run & save ---------------------------

if __name__ == "__main__":
    target_date = datetime.date.today().strftime("%Y-%m-%d")
    df = get_afl_data(target_date)
    if df.is_empty():
        print("No data collected.")
    else:
        out = f"afl_full_pitch_{target_date}.csv"
        df.write_csv(out)
        print(f"✅ Saved {len(df)} rows × {len(df.columns)} cols to {out}")
        print(df.head())
