#!/usr/bin/env python3
"""IPL Analytics Engine — interactive Streamlit dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.styles import inject
from config.settings import DB_PATH, MODELS_DIR
from src.analytics.player_analytics import (
    batting_leaderboard,
    bowling_leaderboard,
    head_to_head_batter_vs_bowler,
    list_players,
    player_form_index,
    player_phase_profile,
)
from src.analytics.team_analytics import (
    chase_defend_profile,
    list_teams,
    match_summary,
    season_run_rate_trends,
    team_h2h_summary,
    team_head_to_head,
    team_season_table,
)
from src.analytics.venue_analytics import list_venues, toss_venue_impact, venue_difficulty_index
from src.ml.match_outcome import predict_chase_win_prob, predict_match_outcome
from src.ml.score_prediction import predict_final_score
from src.utils.db import read_sql, row_count, table_exists

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Source Sans 3, sans-serif", color="#10241c"),
    margin=dict(l=40, r=20, t=50, b=40),
)
COLORWAY = ["#0f3d2e", "#1f7a4d", "#c6f135", "#d9782d", "#245b4a", "#8fbf5a"]


def _require_warehouse() -> bool:
    if not DB_PATH.exists() or not table_exists("deliveries"):
        st.error("Warehouse not found. Run `python scripts/run_etl.py` first.")
        return False
    return True


def _seasons() -> list[int]:
    df = read_sql("SELECT DISTINCT season FROM matches ORDER BY season")
    return df["season"].astype(int).tolist()


def page_overview() -> None:
    balls = row_count("deliveries")
    matches = row_count("matches")
    players = row_count("players")
    seasons = _seasons()

    st.markdown(
        f"""
        <div class="hero">
          <h1>IPL ANALYTICS ENGINE</h1>
          <p>End-to-end cricket intelligence on {balls:,}+ ball-by-ball records —
          ETL pipelines, SQL warehouse, player & venue labs, and ML models for
          match outcome and score prediction.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="metric-strip">
          <div class="metric-chip"><span>Deliveries</span><strong>{balls:,}</strong></div>
          <div class="metric-chip"><span>Matches</span><strong>{matches:,}</strong></div>
          <div class="metric-chip"><span>Players</span><strong>{players:,}</strong></div>
          <div class="metric-chip"><span>Seasons</span><strong>{seasons[0]}–{seasons[-1]}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    season = st.selectbox("Season filter", ["All"] + seasons, index=len(seasons))
    season_val = None if season == "All" else int(season)

    standings = team_season_table(season_val)
    if season_val is None:
        # latest season snapshot
        latest = standings["season"].max()
        standings = standings[standings["season"] == latest]

    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.subheader("Points race")
        fig = px.bar(
            standings.sort_values("wins"),
            x="wins",
            y="team_name",
            orientation="h",
            color="win_pct",
            color_continuous_scale=["#0f3d2e", "#c6f135"],
            labels={"wins": "Wins", "team_name": "", "win_pct": "Win %"},
        )
        fig.update_layout(**PLOTLY_LAYOUT, colorway=COLORWAY, height=420)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Phase run rates over years")
        trends = season_run_rate_trends()
        fig2 = px.line(
            trends,
            x="season",
            y="run_rate",
            color="phase",
            markers=True,
            color_discrete_sequence=COLORWAY,
        )
        fig2.update_layout(**PLOTLY_LAYOUT, height=420)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Recent matches")
    st.dataframe(match_summary(season_val, limit=30), use_container_width=True, hide_index=True)


def page_players() -> None:
    st.header("Player Lab")
    seasons = _seasons()
    season = st.selectbox("Season", ["Career"] + seasons, index=len(seasons))
    season_val = None if season == "Career" else int(season)
    min_balls = st.slider("Minimum balls", 30, 200, 60, 10)

    tab1, tab2, tab3, tab4 = st.tabs(["Batting", "Bowling", "Form Index", "Matchup"])
    with tab1:
        bat = batting_leaderboard(season_val, min_balls=min_balls, limit=40)
        fig = px.scatter(
            bat,
            x="strike_rate",
            y="runs",
            size="balls",
            hover_name="player_name",
            color="sixes",
            color_continuous_scale=["#0f3d2e", "#d9782d"],
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=480)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(bat, use_container_width=True, hide_index=True)

    with tab2:
        bowl = bowling_leaderboard(season_val, min_balls=min_balls, limit=40)
        fig = px.scatter(
            bowl,
            x="economy",
            y="wickets",
            size="balls",
            hover_name="player_name",
            color="average",
            color_continuous_scale=["#c6f135", "#0f3d2e"],
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=480)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(bowl, use_container_width=True, hide_index=True)

    with tab3:
        players = list_players(800)["player_name"].tolist()
        default = players[0] if players else ""
        # prefer a known-looking name if present
        for candidate in players:
            if " " in candidate:
                default = candidate
                break
        player = st.selectbox("Player", players, index=players.index(default) if default in players else 0)
        form = player_form_index(player, last_n_matches=10)
        phase = player_phase_profile(player, season_val)
        if form.empty:
            st.info("No batting innings for this player.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=form["match_date"], y=form["form_index"],
                mode="lines+markers", name="Form", line=dict(color="#1f7a4d", width=3),
            ))
            fig.add_trace(go.Scatter(
                x=form["match_date"], y=form["rolling_form"],
                mode="lines", name="Rolling form", line=dict(color="#d9782d", width=2, dash="dot"),
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=360, title=f"{player} — form index")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(form, use_container_width=True, hide_index=True)
        if not phase.empty:
            figp = px.bar(phase, x="phase", y="strike_rate", color="runs", color_continuous_scale=["#0f3d2e", "#c6f135"])
            figp.update_layout(**PLOTLY_LAYOUT, height=320, title="Phase strike rates")
            st.plotly_chart(figp, use_container_width=True)

    with tab4:
        players = list_players(800)["player_name"].tolist()
        c1, c2 = st.columns(2)
        batter = c1.selectbox("Batter", players, key="h2h_bat")
        bowler = c2.selectbox("Bowler", players, key="h2h_bowl")
        h2h = head_to_head_batter_vs_bowler(batter, bowler)
        st.dataframe(h2h, use_container_width=True, hide_index=True)


def page_venues() -> None:
    st.header("Venue Lab")
    seasons = _seasons()
    season = st.selectbox("Season", ["All"] + seasons, index=0, key="venue_season")
    season_val = None if season == "All" else int(season)
    idx = venue_difficulty_index(season_val)
    st.subheader("Batting friendliness index (100 = league average)")
    fig = px.bar(
        idx.sort_values("batting_index"),
        x="batting_index",
        y="venue_name",
        orientation="h",
        color="chase_friendliness",
        color_continuous_scale=["#0f3d2e", "#d9782d"],
        hover_data=["avg_first_innings", "matches", "city"],
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=520)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(idx, use_container_width=True, hide_index=True)

    st.subheader("Toss conversion by venue")
    venues = ["All"] + list_venues()["venue_name"].tolist()
    venue = st.selectbox("Venue", venues)
    toss = toss_venue_impact(None if venue == "All" else venue)
    fig2 = px.bar(
        toss,
        x="venue_name",
        y="toss_win_convert_pct",
        color="toss_decision",
        barmode="group",
        color_discrete_sequence=COLORWAY,
    )
    fig2.update_layout(**PLOTLY_LAYOUT, height=400, xaxis_tickangle=-35)
    st.plotly_chart(fig2, use_container_width=True)


def page_teams() -> None:
    st.header("Team & Head-to-Head")
    # Prefer teams that actually appear in matches for meaningful H2H
    active = read_sql(
        """
        SELECT DISTINCT t.team_name
        FROM teams t
        JOIN matches m ON t.team_id IN (m.team1_id, m.team2_id)
        ORDER BY t.team_name
        """
    )["team_name"].tolist()
    teams = active or list_teams()["team_name"].tolist()
    seasons = _seasons()
    season = st.selectbox("Season", ["All"] + seasons, index=len(seasons), key="team_season")
    season_val = None if season == "All" else int(season)

    st.subheader("Chase vs defend")
    cdf = chase_defend_profile(season_val)
    fig = px.bar(
        cdf,
        x="team_name",
        y="win_pct",
        color="role",
        barmode="group",
        color_discrete_sequence=["#0f3d2e", "#d9782d"],
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=420, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Head-to-head")
    c1, c2 = st.columns(2)
    a = c1.selectbox("Team A", teams, index=0)
    b_options = [t for t in teams if t != a] or teams
    b = c2.selectbox("Team B", b_options, index=0)
    if a == b:
        st.warning("Pick two different teams.")
        return
    summary = team_h2h_summary(a, b)
    m1, m2, m3 = st.columns(3)
    m1.metric("Meetings", summary["matches"])
    m2.metric(f"{a} wins", summary[a])
    m3.metric(f"{b} wins", summary[b])
    st.dataframe(team_head_to_head(a, b), use_container_width=True, hide_index=True)


def page_predictions() -> None:
    st.header("ML Predictions")
    st.caption("Models: Gradient Boosting / Random Forest trained on warehouse features.")

    has_models = (MODELS_DIR / "match_outcome.joblib").exists()
    if not has_models:
        st.warning("Models not trained yet. Run `python scripts/train_models.py`.")
        return

    with open(MODELS_DIR / "match_outcome_metrics.json") as f:
        m_out = json.load(f)
    with open(MODELS_DIR / "score_prediction_metrics.json") as f:
        m_score = json.load(f)
    with open(MODELS_DIR / "win_probability_metrics.json") as f:
        m_wp = json.load(f)

    c1, c2, c3 = st.columns(3)
    c1.metric("Outcome ROC-AUC", f"{m_out['roc_auc']:.3f}")
    sel = m_score["selected"]
    c2.metric("Score MAE", f"{m_score[sel]['mae']:.2f}")
    c3.metric("Win-prob ROC-AUC", f"{m_wp['roc_auc']:.3f}")

    tab1, tab2, tab3 = st.tabs(["Match outcome", "Score projection", "Live win probability"])

    teams = list_teams()
    venues = list_venues()
    seasons = _seasons()

    with tab1:
        active_teams = read_sql(
            """
            SELECT DISTINCT t.team_id, t.team_name
            FROM teams t
            JOIN matches m ON t.team_id IN (m.team1_id, m.team2_id)
            ORDER BY t.team_name
            """
        )
        team_names = active_teams["team_name"].tolist() or teams["team_name"].tolist()
        c1, c2 = st.columns(2)
        t1 = c1.selectbox("Team 1", team_names, key="pred_t1")
        t2_options = [t for t in team_names if t != t1] or team_names
        t2 = c2.selectbox("Team 2", t2_options, key="pred_t2")
        venue = st.selectbox("Venue", venues["venue_name"], key="pred_venue")
        season = st.selectbox("Season context", seasons, index=len(seasons) - 1, key="pred_season")
        toss = st.radio("Toss winner", ["Team 1", "Team 2"], horizontal=True)
        decision = st.radio("Toss decision", ["field", "bat"], horizontal=True)

        id_map = dict(zip(active_teams["team_name"], active_teams["team_id"])) if not active_teams.empty else dict(zip(teams["team_name"], teams["team_id"]))
        t1_id = int(id_map[t1])
        t2_id = int(id_map[t2])
        v_id = int(venues.loc[venues["venue_name"] == venue, "venue_id"].iloc[0])

        rates = read_sql(
            """
            SELECT team_id, AVG(win) AS win_rate FROM (
              SELECT team1_id AS team_id, CASE WHEN winner_id=team1_id THEN 1.0 ELSE 0.0 END AS win FROM matches WHERE winner_id IS NOT NULL
              UNION ALL
              SELECT team2_id AS team_id, CASE WHEN winner_id=team2_id THEN 1.0 ELSE 0.0 END AS win FROM matches WHERE winner_id IS NOT NULL
            ) GROUP BY team_id
            """
        )
        rate_map = dict(zip(rates["team_id"], rates["win_rate"]))
        vavg = read_sql(
            "SELECT AVG(i.total_runs) AS a FROM innings i JOIN matches m ON m.match_id=i.match_id WHERE i.innings_number=1 AND m.venue_id=:v",
            {"v": v_id},
        )
        venue_avg = float(vavg.iloc[0]["a"] or 165)

        features = {
            "venue_id": v_id,
            "team1_id": t1_id,
            "team2_id": t2_id,
            "team1_won_toss": 1 if toss == "Team 1" else 0,
            "chose_bat": 1 if decision == "bat" else 0,
            "team1_win_rate": float(rate_map.get(t1_id, 0.5)),
            "team2_win_rate": float(rate_map.get(t2_id, 0.5)),
            "win_rate_diff": float(rate_map.get(t1_id, 0.5) - rate_map.get(t2_id, 0.5)),
            "venue_avg_score": venue_avg,
            "season": int(season),
        }
        if st.button("Predict winner", type="primary"):
            out = predict_match_outcome(features)
            fig = go.Figure(
                go.Bar(
                    x=[t1, t2],
                    y=[out["team1_win_probability"] * 100, out["team2_win_probability"] * 100],
                    marker_color=["#0f3d2e", "#d9782d"],
                )
            )
            fig.update_layout(**PLOTLY_LAYOUT, yaxis_title="Win probability %", height=360)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.write("Project final first-innings score from a live checkpoint.")
        over = st.slider("Overs completed", 5, 16, 10)
        cum_runs = st.number_input("Runs so far", 20, 250, 78)
        cum_wickets = st.slider("Wickets down", 0, 9, 2)
        venue_id = int(venues.loc[venues["venue_name"] == st.selectbox("Venue ", venues["venue_name"], key="score_v"), "venue_id"].iloc[0])
        season = st.selectbox("Season ", seasons, index=len(seasons) - 1, key="score_s")
        balls = over * 6
        state = {
            "over_number": over,
            "cum_runs": cum_runs,
            "cum_wickets": cum_wickets,
            "balls_faced": balls,
            "current_rr": cum_runs / max(over, 1),
            "wickets_in_hand": 10 - cum_wickets,
            "overs_left": 20 - over,
            "proj_naive": (cum_runs / max(over, 1)) * 20,
            "venue_id": venue_id,
            "season": int(season),
        }
        if st.button("Project score", type="primary"):
            pred = predict_final_score(state)
            st.metric("Predicted final score", f"{pred:.0f}")
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=pred,
                gauge={
                    "axis": {"range": [80, 260]},
                    "bar": {"color": "#1f7a4d"},
                    "steps": [
                        {"range": [80, 140], "color": "#dfe8d8"},
                        {"range": [140, 180], "color": "#c6f135"},
                        {"range": [180, 260], "color": "#d9782d"},
                    ],
                },
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=300)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.write("Upgrade: chase win probability from match state.")
        target = st.number_input("Target", 100, 280, 180)
        overs_done = st.slider("Overs bowled in chase", 1, 19, 12, key="wp_over")
        runs = st.number_input("Runs scored", 0, 280, 110, key="wp_runs")
        wickets = st.slider("Wickets lost", 0, 9, 3, key="wp_w")
        balls_faced = overs_done * 6
        runs_needed = max(target - runs, 0)
        balls_left = max(120 - balls_faced, 0)
        req_rr = (runs_needed / (balls_left / 6.0)) if balls_left else 99.0
        state = {
            "runs_needed": runs_needed,
            "balls_left": balls_left,
            "wickets_left": 10 - wickets,
            "required_rr": req_rr,
            "cum_runs": runs,
            "over_number": overs_done,
        }
        if st.button("Compute win probability", type="primary"):
            p = predict_chase_win_prob(state)
            # simple curve around current state
            xs = list(range(max(1, overs_done - 5), min(20, overs_done + 6)))
            ys = []
            for o in xs:
                bf = o * 6
                bl = max(120 - bf, 0)
                # assume linear scoring continuation
                proj_runs = runs + (runs / max(overs_done, 1)) * (o - overs_done)
                rn = max(target - proj_runs, 0)
                rr = (rn / (bl / 6.0)) if bl else 99
                ys.append(
                    predict_chase_win_prob(
                        {
                            "runs_needed": rn,
                            "balls_left": bl,
                            "wickets_left": 10 - wickets,
                            "required_rr": rr,
                            "cum_runs": proj_runs,
                            "over_number": o,
                        }
                    )
                    * 100
                )
            st.metric("Chasing side win probability", f"{p*100:.1f}%")
            fig = px.line(x=xs, y=ys, markers=True, labels={"x": "Over", "y": "Win %"})
            fig.update_traces(line_color="#0f3d2e")
            fig.update_layout(**PLOTLY_LAYOUT, height=360)
            st.plotly_chart(fig, use_container_width=True)


def page_sql() -> None:
    st.header("SQL Workbench")
    st.caption("Read-only queries against the analytics warehouse.")
    default = "SELECT * FROM v_match_summary ORDER BY match_date DESC LIMIT 25"
    query = st.text_area("SQL", value=default, height=140)
    if st.button("Run query"):
        banned = ["insert", "update", "delete", "drop", "alter", "attach", "pragma"]
        low = query.lower()
        if any(b in low for b in banned):
            st.error("Only SELECT queries are allowed.")
            return
        try:
            df = read_sql(query)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("Download CSV", df.to_csv(index=False), "query.csv", "text/csv")
        except Exception as exc:
            st.error(f"Query failed: {exc}")


def main() -> None:
    st.set_page_config(
        page_title="IPL Analytics Engine",
        page_icon="I",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject(st)
    st.sidebar.markdown("## IPL Analytics")
    st.sidebar.caption("Python · SQL · Pandas · Scikit-learn · Streamlit · Power BI")
    page = st.sidebar.radio(
        "Navigate",
        [
            "Overview",
            "Player Lab",
            "Venue Lab",
            "Teams & H2H",
            "ML Predictions",
            "SQL Workbench",
        ],
    )
    if not _require_warehouse():
        st.stop()

    pages = {
        "Overview": page_overview,
        "Player Lab": page_players,
        "Venue Lab": page_venues,
        "Teams & H2H": page_teams,
        "ML Predictions": page_predictions,
        "SQL Workbench": page_sql,
    }
    pages[page]()


if __name__ == "__main__":
    main()
