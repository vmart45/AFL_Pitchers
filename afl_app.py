import datetime as dt
from zoneinfo import ZoneInfo
import os
import requests
import pandas as pd
import polars as pl
import streamlit as st
import matplotlib.pyplot as plt
from st_aggrid import AgGrid, GridOptionsBuilder
from PIL import Image
from io import BytesIO
from typing import Optional

st.set_page_config(page_title="AFL Pitch Movement Dashboard", layout="wide")
st.title("AFL Pitcher Dashboard")

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

def get_player_headshot(pitcher_id: str):
    """Fetch MLB headshot image and return PIL Image."""
    try:
        url = (
            f"https://img.mlbstatic.com/mlb-photos/image/"
            f"upload/d_people:generic:headshot:67:current.png/"
            f"w_640,q_auto:best/v1/people/{pitcher_id}/headshot/silo/current.png"
        )
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
    except Exception:
        pass
    return None

@st.cache_data(ttl=86400)
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
    """Load cached AFL data or refresh automatically after 8am PT each day."""
    PT = ZoneInfo("America/Los_Angeles")
    now_pt = dt.datetime.now(PT)

    # Define the daily 8am cutoff
    cutoff = now_pt.replace(hour=8, minute=0, second=0, microsecond=0)
    if now_pt < cutoff:
        cutoff = cutoff - dt.timedelta(days=1)

    if os.path.exists(CACHE_FILE):
        mtime = dt.datetime.fromtimestamp(os.path.getmtime(CACHE_FILE), PT)
        age_hours = (now_pt - mtime).total_seconds() / 3600
        if mtime >= cutoff and age_hours < 12:
            return pl.read_parquet(CACHE_FILE)


    # Otherwise fetch new data
    season_start = dt.date(2025, 10, 1)
    today = dt.date.today()
    yesterday = today - dt.timedelta(days=1)

    st.info(f"Refreshing data (last cache before {cutoff.strftime('%b %d %H:%M PT')})...")
    df = fetch_afl_data_all_days(season_start, today)
    if df.is_empty():
        df = fetch_afl_data_all_days(season_start, yesterday)
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
    "Fastball": "#D22D49",
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
    "Sweeper": "#DDB33A",
    "Curveball": "#00D1ED",
    "Knuckle Curve": "#6236CD",
    "Knuckleball": "#3C44CD",
    "Slow Curve": "#0068FF"
}

def normalize_pitch_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    name_lower = name.lower().strip()
    if "four" in name_lower and "seam" in name_lower:
        return "Fastball"
    if "4-seam" in name_lower or "4 seam" in name_lower:
        return "Fastball"
    if "fourseam" in name_lower:
        return "Fastball"
    return name

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
    
def get_pitcher_bio(pitcher_id: int):
    """Fetch basic bio info from MLB Stats API."""
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json().get("people", [{}])[0]
            full_name = data.get("fullName", "Unknown")
            height = data.get("height", "-")
            weight = data.get("weight", "-")
            birth_date = data.get("birthDate", "-")
            age = data.get("currentAge", None)
            city = data.get("birthCity", "")
            state = data.get("birthStateProvince", "")
            country = data.get("birthCountry", "")
            throws = data.get("pitchHand", {}).get("description", "-")
            bats = data.get("batSide", {}).get("description", "-")

            birthplace = ", ".join([x for x in [city, state, country] if x])

            return {
                "Name": full_name,
                "Throws": throws,
                "Bats": bats,
                "Height": height,
                "Weight": f"{weight} lbs" if isinstance(weight, (int, float)) else weight,
                "Birth Date": birth_date,
                "Age": age,
                "Birthplace": birthplace
            }
    except Exception:
        pass
    return None


def infer_pitcher_team(df):
    """Infer pitcher's team using home/away + is_top_inning (TRUE/FALSE)."""
    if {"home_team", "away_team", "is_top_inning"} <= set(df.columns):
        # Get first row for context
        row = df[0]

        def _unwrap(val):
            """Safely unwrap Series/list/scalar."""
            if isinstance(val, pl.Series):
                return val.item()
            elif isinstance(val, list):
                return val[0] if val else None
            return val

        home = _unwrap(row["home_team"])
        away = _unwrap(row["away_team"])
        is_top = _unwrap(row["is_top_inning"])

        # Normalize boolean/string mix
        if isinstance(is_top, str):
            is_top = is_top.strip().upper() == "TRUE"

        # Apply inning logic
        if is_top is True:
            # Top inning: away hitting → pitcher = home
            return home
        elif is_top is False:
            # Bottom inning: home hitting → pitcher = away
            return away

    return "Unknown"

col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    if not df.is_empty():
        pitcher_id = None
        if "pitcher_id" in df.columns:
            pitcher_id = int(df["pitcher_id"][0])

        # show headshot on far left
        headshot = get_player_headshot(pitcher_id) if pitcher_id else None
        if headshot:
            st.image(headshot, width=120)
        else:
            st.image(
                "https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/"
                "w_640,q_auto:best/v1/people/generic/headshot/silo/current.png",
                width=120,
            )

with col2:
    if not df.is_empty():
        team_name = infer_pitcher_team(df)
        bio = get_pitcher_bio(pitcher_id) if pitcher_id else None

        if bio:
            formatted_date = format_date_pretty(selected_date)

            st.markdown(
                f"<h3 style='text-align:center; margin-bottom:0px; font-weight:700;'>"
                f"{bio['Name']} — {formatted_date}</h3>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<h5 style='text-align:center; color:#555; margin-top:2px; font-weight:600;'>"
                f"{team_name}</h5>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<p style='text-align:center; font-size:14px; margin-top:4px;'>"
                f"<b>Throws/Bats:</b> {bio['Throws']} / {bio.get('Bats', '-')} "
                f"| <b>Height/Weight:</b> {bio['Height']}, {bio['Weight']} "
                f"| <b>Born:</b> {bio['Birthplace']} — {bio['Birth Date']} "
                f"({bio['Age']} yrs old)"
                f"</p>",
                unsafe_allow_html=True
            )


# --- Pitch movement plot ---
if "breakHorizontal" in df.columns and "breakVerticalInduced" in df.columns:
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(3.5, 3.3))

    if "type__description" in df.columns:
        pitch_counts = (
            df.group_by("type__description")
              .count()
              .sort("count", descending=True)
              .to_pandas()
        )
        pitch_types_present = pitch_counts["type__description"].tolist()
    else:
        pitch_types_present = []

    for pt in pitch_types_present:
        g = df.filter(pl.col("type__description") == pt)
        if g.is_empty():
            continue

        pt_norm = normalize_pitch_name(pt)
        color = PITCH_COLORS.get(pt_norm, "gray")

        ax.scatter(
            g["breakHorizontal"],
            g["breakVerticalInduced"],
            label=pt_norm,
            color=color,
            alpha=0.8,
            s=15,
            edgecolors="black",
            linewidths=0.3,
        )

    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")

    # Legend styled like example image
    legend = ax.legend(
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=min(len(pitch_types_present), 4),
        fontsize=6,
        title=None,
        columnspacing=1.0,
        handlelength=1.5,
        handletextpad=0.4,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("lightgray")
    legend.get_frame().set_linewidth(0.5)

    formatted_date = format_date_pretty(selected_date)
    ax.set_title(
        f"Pitch Movement",
        fontsize=8, fontweight="bold", pad=8
    )
    ax.set_xlabel("Horizontal Break (in.)", fontsize=7, labelpad=6)
    ax.set_ylabel("Induced Vertical Break (in.)", fontsize=7, labelpad=6)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(left=False, labelleft=False, bottom=True, labelbottom=True)

    plt.tight_layout(pad=1)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.pyplot(fig, clear_figure=True, use_container_width=False)


def format_feet_inches(decimal_value):
    if decimal_value is None or pl.Series([decimal_value]).is_null().any():
        return "-"
    feet = int(decimal_value)
    inches = round((decimal_value - feet) * 12)
    return f"{feet}′{inches}″"


def pitch_table(df, ax, fontsize: int = 8):
    # rename + reorder columns
    df = df.rename(columns={
        "type__description": "Pitch",
        "Count": "Count",
        "Mix%": "Mix%",
        "Velo": "Velo",
        "Spin": "Spin",
        "IVB": "IVB",
        "HB": "HB",
        "RelHt": "RelHt",
        "Ext": "Ext"
    })[["Pitch", "Count", "Mix%", "Velo", "Spin", "IVB", "HB", "RelHt", "Ext"]]

    # round numeric columns
    for col in ["Velo", "IVB", "HB", "Mix%"]:
        if col in df.columns:
            df[col] = df[col].round(1)
    if "Spin" in df.columns:
        df["Spin"] = df["Spin"].round(0)
    if "Mix%" in df.columns:
        df["Mix%"] = df["Mix%"].astype(str) + "%"

    table_plot = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1]
    )

    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(fontsize)
    table_plot.scale(1, 0.55)

    for (row, col), cell in table_plot.get_celld().items():
        if row == 0:
            # header row
            cell.set_facecolor("#FFFFFF")
            cell.set_text_props(weight="bold", color="#000000")
        else:
            if col == 0:
                pitch_name = cell.get_text().get_text()
                color = PITCH_COLORS.get(pitch_name, "#999999")
                cell.set_facecolor(color)
                cell.set_text_props(weight="bold", color="#FFFFFF", fontsize=fontsize)
            else:
                cell.set_facecolor("#FFFFFF")
                cell.set_text_props(color="#000000", fontsize=fontsize)

    ax.axis("off")
    return ax
    
# --- Summary table (single, robust block) ---
# --- Pitch summary table ---
if not df.is_empty():
    total_pitches = df.height

    # choose best available release-height column
    rel_ht_col = (
        "z0" if "z0" in df.columns
        else ("releasePosZ" if "releasePosZ" in df.columns else None)
    )

    agg_exprs = [
        pl.count().alias("Count"),
        (pl.count() / total_pitches * 100).alias("Mix%"),
    ]
    if "startSpeed" in df.columns:
        agg_exprs.append(pl.col("startSpeed").mean().alias("Velo"))
    if "spinRate" in df.columns:
        agg_exprs.append(pl.col("spinRate").mean().alias("Spin"))
    if "breakVerticalInduced" in df.columns:
        agg_exprs.append(pl.col("breakVerticalInduced").mean().alias("IVB"))
    if "breakHorizontal" in df.columns:
        agg_exprs.append(pl.col("breakHorizontal").mean().alias("HB"))
    if rel_ht_col:
        agg_exprs.append(pl.col(rel_ht_col).mean().alias("RelHt"))
    if "extension" in df.columns:
        agg_exprs.append(pl.col("extension").mean().alias("Ext"))

    summary = (
        df.group_by("type__description")
          .agg(agg_exprs)
          .sort("Count", descending=True)
    )

if summary.height > 0:
    df_summary = summary.to_pandas()
    df_summary["type__description"] = df_summary["type__description"].apply(normalize_pitch_name)

    # Round and format
    for col in ["Velo", "IVB", "HB"]:
        if col in df_summary.columns:
            df_summary[col] = df_summary[col].round(1)
    if "Spin" in df_summary.columns:
        df_summary["Spin"] = df_summary["Spin"].round(0)
    if "Mix%" in df_summary.columns:
        df_summary["Mix%"] = df_summary["Mix%"].round(1)

    for col in ["RelHt", "Ext"]:
        if col in df_summary.columns:
            df_summary[col] = df_summary[col].apply(lambda v: f"{int(v)}′{round((v-int(v))*12)}″" if pd.notna(v) else "-")

    # Larger table scaling
    fig2, ax2 = plt.subplots(figsize=(8.5, 1.9))  # wider table
    pitch_table(df_summary, ax2, fontsize=8)

    table = ax2.tables[0] if hasattr(ax2, "tables") and ax2.tables else None
    if table:
        for key, cell in table.get_celld().items():
            if key[0] == 0:
                cell.set_height(0.07)  # tighten header
            cell.set_width(0.16)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig2, clear_figure=True, use_container_width = False)
