import datetime as dt
from zoneinfo import ZoneInfo
import os
import polars as pl
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="AFL Pitch Movement Dashboard", layout="wide")
st.title("🎯 Arizona Fall League Pitch Movement Dashboard")


CACHE_FILE = "afl_cache.parquet"

def seconds_until_next_8am_pt(now_utc: dt.datetime | None = None) -> int:
    if now_utc is None:
        now_utc = dt.datetime.now(dt.timezone.utc)
    PT = ZoneInfo("America/Los_Angeles")
    now_pt = now_utc.astimezone(PT)
    target = now_pt.replace(hour=8, minute=0, second=0, microsecond=0)
    if now_pt >= target:
        target = target + dt.timedelta(days=1)
    return int((target - now_pt).total_seconds())

@st.cache_data(ttl=seconds_until_next_8am_pt())
def fetch_afl_data_all_days(start: dt.date, end: dt.date) -> pl.DataFrame:
    dfs = []
    all_cols = set()
    for i in range((end - start).days + 1):
        date = start + dt.timedelta(days=i)
        df_day = get_afl_data(date.strftime("%Y-%m-%d"))
        if not df_day.is_empty():
            dfs.append(df_day)
            all_cols.update(df_day.columns)
    if not dfs:
        return pl.DataFrame()
    all_cols = sorted(all_cols)
    aligned = []
    for df in dfs:
        missing = set(all_cols) - set(df.columns)
        if missing:
            df = df.with_columns([pl.lit(None).alias(col) for col in missing])
        aligned.append(df.select(all_cols))
    return pl.concat(aligned, how="vertical", rechunk=True)

def load_or_fetch_data() -> pl.DataFrame:
    PT = ZoneInfo("America/Los_Angeles")
    now_pt = dt.datetime.now(PT)
    cutoff = now_pt.replace(hour=8, minute=0, second=0, microsecond=0)
    if now_pt < cutoff:
        cutoff = cutoff - dt.timedelta(days=1)
    if os.path.exists(CACHE_FILE):
        mtime = dt.datetime.fromtimestamp(os.path.getmtime(CACHE_FILE), PT)
        if mtime >= cutoff:
            return pl.read_parquet(CACHE_FILE)
    season_start = dt.date(2025, 10, 1)
    today = dt.date.today()
    df = fetch_afl_data_all_days(season_start, today)
    if not df.is_empty():
        df.write_parquet(CACHE_FILE)
    return df

AFL_TEAMS = [
    "Glendale Desert Dogs",
    "Mesa Solar Sox",
    "Peoria Javelinas",
    "Salt River Rafters",
    "Scottsdale Scorpions",
    "Surprise Saguaros"
]

PITCH_COLORS = {
    "Four-Seam Fastball": "#D22D49",
    "Two-Seam Fastball": "#DE6A04",
    "Sinker": "#FE9D00",
    "Cutter": "#933F2C",
    "Changeup": "#1DBE3A",
    "Splitter": "#3BACAC",
    "Screwball": "#60DB33",
    "Forkball": "#55CCAB",
    "Slurve": "#DDB33A",
    "Slider": "#EEE716",
    "Gyroball": "#FFFF99",
    "Sweeper": "#93AFD4",
    "Curveball": "#00D1ED",
    "Knuckle Curve": "#6236CD",
    "Knuckleball": "#3C44CD",
    "Slow Curve": "#0068FF"
}

def format_date_pretty(date_str: str) -> str:
    date = dt.datetime.strptime(str(date_str), "%Y-%m-%d").date()
    day = date.day
    suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return date.strftime(f"%b. {day}{suffix}, %Y")

@st.cache_data(ttl=86400)
def load_afl_data():
    start = datetime.date(2025, 10, 1)
    end = datetime.date.today()

    dfs = []
    all_cols = set()

    for i in range((end - start).days + 1):
        date = start + datetime.timedelta(days=i)
        df_day = get_afl_data(date.strftime("%Y-%m-%d"))
        if not df_day.is_empty():
            dfs.append(df_day)
            all_cols.update(df_day.columns)

    if not dfs:
        return pl.DataFrame()

    all_cols = sorted(all_cols)
    dfs_aligned = []
    for df in dfs:
        missing = set(all_cols) - set(df.columns)
        if missing:
            for col in missing:
                df = df.with_columns(pl.lit(None).alias(col))
        df = df.select(all_cols)
        dfs_aligned.append(df)

    df = pl.concat(dfs_aligned)
    return df


def load_or_fetch_data():
    """Loads cached data or fetches new AFL data if cache missing."""
    if os.path.exists(CACHE_FILE):
        df = pl.read_parquet(CACHE_FILE)
    else:
        df = load_afl_data()
        if not df.is_empty():
            df.write_parquet(CACHE_FILE)
    return df

df = load_or_fetch_data()

if not df.is_empty() and "home_team" in df.columns:
    df = df.filter(pl.col("home_team").is_in(AFL_TEAMS))

if df.is_empty():
    st.warning("No Arizona Fall League pitch data found.")
    st.stop()


st.sidebar.header("Filters")

# Pitcher
pitchers = sorted(df["pitcher_name"].drop_nulls().unique().to_list())
selected_pitcher = st.sidebar.selectbox("Select Pitcher", pitchers)
df = df.filter(pl.col("pitcher_name") == selected_pitcher)

# Game date
dates = sorted(df["game_date"].unique().to_list())
selected_date = st.sidebar.selectbox("Select Game Date", dates)
df = df.filter(pl.col("game_date") == selected_date)

# Pitch types
if "type__description" in df.columns:
    pitch_types = sorted(df["type__description"].drop_nulls().unique().to_list())
    selected_pitch_types = st.sidebar.multiselect(
        "Select Pitch Type(s)", pitch_types, default=pitch_types
    )
    df = df.filter(pl.col("type__description").is_in(selected_pitch_types))
else:
    st.warning("Pitch type column not found.")
    st.stop()

if "breakHorizontal" in df.columns and "breakVerticalInduced" in df.columns:
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(4, 3.5))  # much smaller footprint

    for group_tuple in df.group_by("type__description"):
        pitch_type = group_tuple[0]
        if isinstance(pitch_type, tuple):
            pitch_type = pitch_type[0]
        group = group_tuple[1]

        color = PITCH_COLORS.get(pitch_type, "gray")
        ax.scatter(
            group["breakHorizontal"],
            group["breakVerticalInduced"],
            label=pitch_type,
            color=color,
            alpha=0.8,
            s=15,
            edgecolors="black",
            linewidths=0.3
        )

    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")

    # Compact legend: place outside, smaller text
    ax.legend(
        frameon=False,
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        fontsize=5,
        title="Pitch Type",
        title_fontsize=6
    )

    formatted_date = format_date_pretty(selected_date)
    ax.set_title(
        f"Pitch Movement — {selected_pitcher} ({formatted_date})",
        fontsize=8,
        fontweight="bold",
        pad=8
    )
    ax.set_xlabel("Horizontal Break (in.)", fontsize=7, labelpad=6)
    ax.set_ylabel("Induced Vertical Break (in.)", fontsize=7, labelpad=6)
    ax.grid(True, linestyle="--", alpha=0.3)

    st.pyplot(fig, clear_figure=True)
else:
    st.warning("Missing breakHorizontal or breakVerticalInduced columns for plotting.")
