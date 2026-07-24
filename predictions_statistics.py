import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
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


# Step 4: Filter last month
today = pd.Timestamp.today().normalize()
last_month = today - pd.DateOffset(days=30)

final_Full = final_Full[final_Full["Date"] >= last_month]

# Outcome arrives from update_results.py as the literal strings 'TRUE',
# 'FALSE', or '' (unmatched). Python treats every non-empty string as
# truthy -- including the string 'FALSE' -- so code further down that did
# `1 if x else -1` was scoring every loss as a win. Convert to a real
# boolean here, once, so every downstream mean/sum/apply is correct.
final_Full['Outcome'] = final_Full['Outcome'].map({'TRUE': True, 'FALSE': False})
# Rows that couldn't be matched (no result yet, data gap) are neither a win
# nor a loss -- drop them rather than let them silently corrupt every
# accuracy/ROI figure below.
final_Full = final_Full.dropna(subset=['Outcome'])
final_Full['Outcome'] = final_Full['Outcome'].astype(bool)


# ---------- Cumulative Growth as Candlestick ----------
final_Full['ResultValue'] = final_Full['Outcome'].apply(lambda x: 1 if x else -1)
final_Full['Cumulative'] = final_Full['ResultValue'].cumsum()

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
    increasing_line_color="#9CFF00",   # neon green (growth)
    decreasing_line_color="#FF6B6B",   # soft red (decline)
    increasing_fillcolor="#9CFF00",
    decreasing_fillcolor="#FF6B6B"
)])

fig_candle.add_hline(y=0, line=dict(color="rgba(255,255,255,0.3)", dash="dot"))

fig_candle.update_layout(
    title=dict(
        text="📈 Cumulative Growth of Predictions",
        font=dict(size=22, color="white"),
        x=0.5,
        xanchor="center"
    ),
    xaxis=dict(
        title="Date",
        color="white",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.15)"
    ),
    yaxis=dict(
        title="Cumulative Score",
        color="white",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.15)"
    ),
    paper_bgcolor="rgba(0,0,0,0)",  # fully transparent
    plot_bgcolor="rgba(0,0,0,0)",   # fully transparent
    font=dict(family="Arial", size=14, color="white"),
    hovermode="x unified"
)
fig_candle.show()
fig_candle.write_image("pics/plot1.png", scale=3)

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
    color_discrete_map={"Home": "#1ABC9C", "Away": "#9B59B6"}, 
    symbol_map={"Home": "circle", "Away": "diamond"}  
)

fig_combined.update_traces(
    opacity=0.9,
    marker=dict(size=13)  # no outline
)

fig_combined.update_yaxes(range=[-0.05, 1.05])

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


# --- Free vs VIP Accuracy Area
acc_trend = (
    final_Full.groupby(["Date", "PickedforFree"])
    .Outcome.mean()
    .reset_index()
)

acc_trend["Type"] = acc_trend["PickedforFree"].map({True: "Free", False: "VIP"})

vip_fill = "rgba(211,156,65,0.2)"   # VIP gold, 20% opacity
free_fill = "rgba(4,34,69,0.2)"     # Free dark blue, 20% opacity

fig_weekly = go.Figure()

for t, line_color, fill_color in [("VIP", "#D39C41", "rgba(211,156,65,0.2)"),
                                  ("Free", "#1F528B", "rgba(4,34,69,0.2)")]:
    df_plot = acc_trend[acc_trend["Type"] == t]
    fig_weekly.add_trace(go.Scatter(
        x=df_plot["Date"],
        y=df_plot["Outcome"],
        mode="lines+markers",
        name=t,
        line=dict(color=line_color, width=2),
        marker=dict(size=6),
        fill='tozeroy',      # fill area under the line
        fillcolor=fill_color
    ))

fig_weekly.update_yaxes(range=[-0.05, 1.05], tickformat=".0%")
fig_weekly.update_layout(
    title="📈 Free vs VIP Accuracy Over Time",
    template="plotly_dark",
    plot_bgcolor="#111111",
    paper_bgcolor="#111111",
    font=dict(family="Arial", size=14, color="white"),
    legend=dict(title="Prediction Type"),
    hovermode="x unified"
)

fig_weekly.update_xaxes(
    tickformat="%b %d %Y",   # only show date, no hours
    #tickangle=45              # optional: tilt for readability
)

fig_weekly.show()


# --- Division Bars (Free Picks Performance)
division_perf = (
    final_Full.groupby(["Division", "PickedforFree"])
    .Outcome.mean()
    .reset_index()
)

division_perf["Type"] = division_perf["PickedforFree"].map({True: "Free", False: "VIP"})

# Sort divisions by VIP accuracy (or overall mean)
division_order = (
    division_perf.groupby("Division")["Outcome"].mean()
    .sort_values(ascending=True)
    .index.tolist()
)
fig_bars = px.bar(
    division_perf,
    x="Outcome",
    y="Division",
    color="Type",
    orientation="h",
    barmode="group",
    category_orders={"Division": division_order},
    labels={"Outcome": "Accuracy", "Division": "Division", "Type": "Prediction Type"},
    title="⚽ Accuracy by Division – Free vs VIP",
    color_discrete_map={"Free": "#042245", "VIP": "#D39C41"}  # friendly colors
)

fig_bars.update_traces(
    text=division_perf["Outcome"].apply(lambda v: f"{v:.0%}"),
    textposition="outside",
    hovertemplate="%{y} – %{color}<br>Accuracy: %{x:.0%}<extra></extra>"
)

fig_bars.update_layout(
    template="plotly_dark",
    plot_bgcolor="#111111",
    paper_bgcolor="#111111",
    font=dict(family="Arial", size=14, color="white"),
    xaxis=dict(title="Accuracy (%)", tickformat=".0%"),
    yaxis=dict(title=None),
    legend=dict(title="Prediction Type")
)

fig_bars.show()


# --- VIP vs Free cumulative performance
def cumulative_curve(df):
    df = df.copy().sort_values("Date")
    df["Cumulative"] = df["ResultValue"].cumsum()
    return df[["Date", "Cumulative"]]

vip_curve = cumulative_curve(final_Full[final_Full["PickedforFree"] == False])
free_curve = cumulative_curve(final_Full[final_Full["PickedforFree"] == True])

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=("💎 VIP Predictions", "📢 Free Predictions")
)

fig.add_trace(
    go.Scatter(
        x=vip_curve["Date"], y=vip_curve["Cumulative"],
        mode="lines",
        line=dict(color="#D39C41", width=4),
        name="VIP"
    ), row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=free_curve["Date"], y=free_curve["Cumulative"],
        mode="lines",
        line=dict(color="#1F528B", width=4),
        name="Free"
    ), row=2, col=1
)

fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.3)", dash="dot"), row=1, col=1)
fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.3)", dash="dot"), row=2, col=1)

fig.update_layout(
    title=dict(
        text="📊 Cumulative Growth – VIP vs Free Predictions",
        font=dict(size=22, color="white"),
        x=0.5, xanchor="center"
    ),
    paper_bgcolor="rgba(0,0,0,0)",  # transparent for IG
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Arial", size=14, color="white"),
    hovermode="x unified",
    height=800
)

fig.update_yaxes(
    title="Cumulative Score",
    color="white",
    showgrid=True,
    gridcolor="rgba(255,255,255,0.15)",
    row=1, col=1
)
fig.update_yaxes(
    title="Cumulative Score",
    color="white",
    showgrid=True,
    gridcolor="rgba(255,255,255,0.15)",
    row=2, col=1
)
fig.update_xaxes(
    title="Date",
    color="white",
    showgrid=True,
    gridcolor="rgba(255,255,255,0.15)",
    tickformat="%b %d %Y",   # e.g. Sep 18 2025
    row=2, col=1
)
fig.write_image("pics/plot2.png", scale=3)
fig.show()


# --- ROI
subscription_per_month = 8
stake = 5
final_Full["Profit"] = final_Full["Outcome"].apply(lambda x: stake if x else -stake)
roi = final_Full.groupby("PickedforFree")["Profit"].sum().reset_index()
roi["Type"] = roi["PickedforFree"].map({True:"Free", False:"VIP"})
roi["ROI_per_€1"] = roi["Profit"] / subscription_per_month

fig_roi = px.bar(
    roi, x="Type", y="ROI_per_€1",
    color="Type",
    text=roi["ROI_per_€1"].apply(lambda x: f"{x:.1f}x"),
    color_discrete_map={"VIP":"#D39C41","Free":"#042245"},
    title="📊 ROI per €1 of Subscription"
)

fig_roi.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",  # transparent for IG
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Arial", size=14, color="white"),
    yaxis=dict(
        title="Profit / €1 Subscription",
        color="white",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.15)"
    ),
    xaxis=dict(
        title="Prediction Type",
        color="white",
        showgrid=False
    ),
    showlegend=False
)

fig_roi.update_traces(
    textposition="outside", 
    insidetextanchor="end",
    textfont=dict(color="white", size=12)
)
fig_roi.show()
fig_roi.write_image("pics/plot3.png", scale=3)
pass