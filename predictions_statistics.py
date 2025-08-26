import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import glob
import os

# ========== USER INPUT ==========
DATA_PATH = "publish/"  # folder where your Excel files live
# =================================

# Step 1: Load all Excel files matching pattern
all_files = glob.glob(os.path.join(DATA_PATH, "update_VIP_*.xlsx"))

dfs = []
for f in all_files:
    df = pd.read_excel(f)
    dfs.append(df)

# Step 2: Concatenate into single DataFrame
final_Full = pd.concat(dfs, ignore_index=True)

# Step 3: Ensure Date is datetime 
final_Full['Date'] = pd.to_datetime(final_Full['Date'], dayfirst=True)


# ---------- Cumulative Growth as Candlestick ----------
final_Full['ResultValue'] = final_Full['Outcome'].apply(lambda x: 1 if x else -1)
final_Full['Cumulative'] = final_Full['ResultValue'].cumsum()

# Resample per date for candlestick (if multiple games per day)
daily = final_Full.groupby("Date").agg(
    Open=("Cumulative", "first"),
    High=("Cumulative", "max"),
    Low=("Cumulative", "min"),
    Close=("Cumulative", "last")
).reset_index()

fig_candle = go.Figure(data=[go.Candlestick(
    x=daily['Date'],
    open=daily['Open'],
    high=daily['High'],
    low=daily['Low'],
    close=daily['Close'],
    increasing_line_color="#00C49A",  # green
    decreasing_line_color="#FF4D6D",  # red
    increasing_fillcolor="#00C49A",
    decreasing_fillcolor="#FF4D6D"
)])

fig_candle.add_hline(y=0, line=dict(color="rgba(255,255,255,0.3)", dash="dot"))

fig_candle.update_layout(
    title="📈 Cumulative Growth of Predictions (Candlestick Style)",
    xaxis_title="Date",
    yaxis_title="Cumulative Score",
    template="plotly_dark",
    plot_bgcolor="#111111",
    paper_bgcolor="#111111",
    font=dict(family="Arial", size=14, color="white"),
    hovermode="x unified"
)
fig_candle.show()


# ---------- Scatter plot for teams ----------
home_stats = final_Full.groupby("HomeTeam").agg(
    Accuracy=("Outcome", "mean"),
    Games=("Outcome", "count")
).reset_index()
home_stats = home_stats[home_stats['Games'] >= 5]
home_stats["Type"] = "Home"
home_stats.rename(columns={"HomeTeam": "Team"}, inplace=True)

away_stats = final_Full.groupby("AwayTeam").agg(
    Accuracy=("Outcome", "mean"),
    Games=("Outcome", "count")
).reset_index()
away_stats = away_stats[away_stats['Games'] >= 5]
away_stats["Type"] = "Away"
away_stats.rename(columns={"AwayTeam": "Team"}, inplace=True)

team_stats = pd.concat([home_stats, away_stats])

# --- Aggregate duplicate positions ---
team_stats_agg = (
    team_stats.groupby(["Games", "Accuracy", "Type"], as_index=False)
    .agg({
        "Team": lambda x: " + ".join(sorted(x)),  # merge team names alphabetically
    })
)

fig_combined = px.scatter(
    team_stats_agg,
    x="Games",
    y="Accuracy",
    size="Games",
    color="Type",
    symbol="Type",
    text=None,           # no label on chart
    hover_name="Team",   # show merged team names on hover
    hover_data={"Games": True, "Accuracy": ":.0%"},
    title="⚽ Home vs Away Teams - Accuracy vs Games",
    labels={"Games": "Number of Predictions", "Accuracy": "Accuracy (%)"},
    color_discrete_map={"Home": "#00C49A", "Away": "#00BFFF"}
)

fig_combined.update_traces(
    opacity=0.85,
    marker=dict(line=dict(width=1, color="white"))
)


fig_combined.update_yaxes(range=[-0.05, 1.05])

# subtle 50% reference line
fig_combined.add_hline(
    y=0.5,
    line=dict(color="lightgray", width=1, dash="dot"),
    opacity=0.5
)

fig_combined.update_layout(
    template="plotly_dark",
    plot_bgcolor="#111111",
    paper_bgcolor="#111111",
    font=dict(family="Arial", size=14, color="white"),
    legend=dict(title="Prediction Type")
)

fig_combined.show()
pass