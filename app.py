"""
app.py - Streamlit Interactive GUI for FIRE Model (Type-Safe Pipeline)
Run via: streamlit run app.py
"""
import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from engine import FIRERunner, DEFAULT_FILE_PATH

st.set_page_config(
    page_title="FIRE Monte Carlo Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔥 FIRE Portfolio Monte Carlo Simulator")


def dict_to_actuals_df(actuals_dict: dict) -> pd.DataFrame:
    rows = []
    for step, val in actuals_dict.items():
        if step == 0:
            y, q = 1, 0
        else:
            y = ((step - 1) // 4) + 1
            q = ((step - 1) % 4) + 1

        rows.append({
            "Year": int(y),
            "Quarter": int(q),
            "Actual Net Worth ($)": float(val)
        })
    return pd.DataFrame(rows)


def actuals_df_to_dict(df: pd.DataFrame) -> dict:
    res = {}
    for _, row in df.iterrows():
        if pd.notna(row["Year"]) and pd.notna(row["Quarter"]) and pd.notna(row["Actual Net Worth ($)"]):
            y = int(row["Year"])
            q = int(row["Quarter"])

            if y == 1 and q == 0:
                step = 0
            elif y >= 1 and 1 <= q <= 4:
                step = ((y - 1) * 4) + q
            else:
                continue
            res[step] = float(row["Actual Net Worth ($)"])
    return res


# --- INITIALIZE SESSION STATE & AUTO-LOAD ---
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    loaded_runner = FIRERunner.load_from_file(DEFAULT_FILE_PATH)

    if loaded_runner is not None:
        st.session_state["salary_1"] = float(loaded_runner.salary_1)
        st.session_state["salary_2"] = float(loaded_runner.salary_2)

        total_income = st.session_state["salary_1"] + st.session_state["salary_2"]
        if loaded_runner.annual_savings_amount is not None:
            st.session_state["annual_savings"] = float(loaded_runner.annual_savings_amount)
            st.session_state["savings_rate_pct"] = (
                        float(loaded_runner.annual_savings_amount) / total_income * 100.0) if total_income > 0 else 30.0
            st.session_state["contrib_mode"] = "Fixed Annual Amount ($)"
        elif loaded_runner.savings_rate is not None:
            st.session_state["savings_rate_pct"] = float(loaded_runner.savings_rate) * 100.0
            st.session_state["annual_savings"] = total_income * float(loaded_runner.savings_rate)
            st.session_state["contrib_mode"] = "Savings Rate (%)"

        st.session_state["wage_growth"] = float(loaded_runner.income_growth_rate) * 100.0
        st.session_state["mean_return"] = float(loaded_runner.mean_return) * 100.0
        st.session_state["volatility"] = float(loaded_runner.volatility) * 100.0
        st.session_state["num_simulations"] = int(loaded_runner.num_simulations)
        st.session_state["seed"] = int(loaded_runner.seed)
        st.session_state.actuals_data = dict_to_actuals_df(loaded_runner.actuals)
        st.toast(f"Loaded state from '{DEFAULT_FILE_PATH}'", icon="📂")
    else:
        st.session_state["salary_1"] = 160_000.0
        st.session_state["salary_2"] = 140_000.0
        st.session_state["annual_savings"] = 90_000.0
        st.session_state["savings_rate_pct"] = 30.0
        st.session_state["contrib_mode"] = "Fixed Annual Amount ($)"
        st.session_state["wage_growth"] = 3.0
        st.session_state["mean_return"] = 7.0
        st.session_state["volatility"] = 16.0
        st.session_state["num_simulations"] = 5_000
        st.session_state["seed"] = 42
        st.session_state.actuals_data = pd.DataFrame({
            "Year": [1, 1, 1, 1, 1],
            "Quarter": [0, 1, 2, 3, 4],
            "Actual Net Worth ($)": [250_000.0, 268_000.0, 290_000.0, 315_000.0, 340_000.0]
        })


def on_contrib_mode_change():
    total_income = float(st.session_state.salary_1 + st.session_state.salary_2)
    if total_income <= 0:
        return

    if st.session_state.contrib_mode == "Savings Rate (%)":
        st.session_state.savings_rate_pct = min(100.0,
                                                max(0.0, (float(st.session_state.annual_savings) / total_income) * 100.0))
    else:
        st.session_state.annual_savings = total_income * (float(st.session_state.savings_rate_pct) / 100.0)


def on_salary_change():
    total_income = float(st.session_state.salary_1 + st.session_state.salary_2)
    if total_income <= 0:
        return

    if st.session_state.contrib_mode == "Fixed Annual Amount ($)":
        st.session_state.savings_rate_pct = min(100.0,
                                                max(0.0, (float(st.session_state.annual_savings) / total_income) * 100.0))
    else:
        st.session_state.annual_savings = total_income * (float(st.session_state.savings_rate_pct) / 100.0)


# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("💾 Persistence")

st.sidebar.header("1. Household & Contributions")
salary_1 = float(st.sidebar.number_input("Salary 1 ($)", step=5_000.0, key="salary_1", on_change=on_salary_change))
salary_2 = float(st.sidebar.number_input("Salary 2 ($)", step=5_000.0, key="salary_2", on_change=on_salary_change))

contrib_mode = st.sidebar.radio(
    "Contribution Mode",
    ["Fixed Annual Amount ($)", "Savings Rate (%)"],
    key="contrib_mode",
    on_change=on_contrib_mode_change
)

total_gross = salary_1 + salary_2

if contrib_mode == "Fixed Annual Amount ($)":
    annual_savings = float(st.sidebar.number_input(
        "Annual Savings ($)",
        step=2_500.0,
        key="annual_savings",
        on_change=on_salary_change
    ))
    savings_rate = None
    implied_pct = (annual_savings / total_gross * 100.0) if total_gross > 0 else 0.0
    st.sidebar.caption(f"Equivalent Savings Rate: **{implied_pct:.1f}%** of gross")
else:
    savings_rate_pct = float(st.sidebar.slider(
        "Savings Rate (%)",
        min_value=1.0,
        max_value=90.0,
        key="savings_rate_pct",
        on_change=on_salary_change
    ))
    savings_rate = savings_rate_pct / 100.0
    annual_savings = None
    implied_amt = total_gross * savings_rate
    st.sidebar.caption(f"Equivalent Annual Contribution: **${implied_amt:,.0f}/yr** (scales with wage growth)")

wage_growth = float(st.sidebar.slider("Annual Wage Growth (%)", min_value=0.0, max_value=10.0, key="wage_growth")) / 100.0

st.sidebar.header("2. Market Parameters")
mean_return = float(st.sidebar.slider("Expected Real Return (%)", min_value=1.0, max_value=12.0, key="mean_return")) / 100.0
volatility = float(st.sidebar.slider("Annual Volatility (%)", min_value=0.0, max_value=30.0, key="volatility")) / 100.0

st.sidebar.header("3. Simulation Engine")
total_horizon = int(st.sidebar.slider("Simulation Horizon (Years)", min_value=5, max_value=40, value=15))

sim_options = [1_000, 5_000, 10_000, 25_000, 50_000]
num_sims = int(st.sidebar.selectbox("Number of Runs", sim_options, key="num_simulations"))
seed = int(st.sidebar.number_input("Random Seed", key="seed"))

# --- MAIN CONTENT LAYOUT ---
col1, col2 = st.columns([1.2, 2.3])

with col1:
    st.subheader("Check-ins Data")
    st.caption("Input real-world net worth check-ins.")

    edited_actuals_df = st.data_editor(
        st.session_state.actuals_data,
        num_rows="dynamic",
        key="actuals_editor",
        width='stretch',
        column_config={
            "Year": st.column_config.NumberColumn("Year", min_value=1, max_value=50, step=1),
            "Quarter": st.column_config.NumberColumn("Quarter", min_value=0, max_value=4, step=1),
            "Actual Net Worth ($)": st.column_config.NumberColumn("Actual Net Worth ($)", step=1000.0)
        }
    )

actuals_dict = actuals_df_to_dict(edited_actuals_df)

runner = FIRERunner(
    salary_1=salary_1,
    salary_2=salary_2,
    annual_savings_amount=annual_savings,
    savings_rate=savings_rate,
    income_growth_rate=wage_growth,
    mean_return=mean_return,
    volatility=volatility,
    num_simulations=num_sims,
    seed=seed,
    actuals=actuals_dict
)

if st.sidebar.button("💾 Save State to 'fire_profile.json'"):
    runner.save_to_file(DEFAULT_FILE_PATH)
    st.sidebar.success(f"Saved directly to '{DEFAULT_FILE_PATH}'!")

try:
    results_df, paths_df = runner.run_simulation(
        total_horizon_years=total_horizon,
        num_simulations=num_sims,
        seed=seed
    )

    with col2:
        st.subheader("Portfolio Net Worth Projection")

        fig = go.Figure()

        # 1. Background Sample Runs
        show_bg = st.checkbox("Show background Monte Carlo trajectories", value=True)
        if show_bg:
            bg_cols = [c for c in paths_df.columns if c != "Year_Float"]
            for col_name in bg_cols:
                valid_mask = paths_df[col_name].notna()
                fig.add_trace(go.Scatter(
                    x=paths_df["Year_Float"][valid_mask],
                    y=paths_df[col_name][valid_mask],
                    mode="lines",
                    line=dict(color="rgba(120, 120, 120, 0.12)", width=1),
                    hoverinfo="skip",
                    showlegend=False
                ))

        # 2. Shaded Confidence Band (P10 to P90)
        valid_p = results_df["P10_Bear"].notna()
        fig.add_trace(go.Scatter(
            x=results_df["Year_Float"][valid_p],
            y=results_df["P90_Bull"][valid_p],
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=results_df["Year_Float"][valid_p],
            y=results_df["P10_Bear"][valid_p],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0, 150, 255, 0.15)",
            name="10th-90th Percentile Range",
            hoverinfo="skip"
        ))

        hover_data_custom = np.stack((results_df["Year_Label"], results_df.index), axis=-1)

        # 3. TRACES ORDERED STRICTLY: Green (P90) -> Blue (P50) -> Red (P10)

        # A. Green (Bull)
        fig.add_trace(go.Scatter(
            x=results_df["Year_Float"][valid_p],
            y=results_df["P90_Bull"][valid_p],
            mode="lines",
            line=dict(color="#00e676", width=2, dash="dash"),
            name="90th Percentile (Bull)",
            customdata=hover_data_custom[valid_p],
            text=results_df["Year_Label"][valid_p],
            hovertemplate="%{text}: %{y:$.3s}<extra></extra>"
        ))

        # B. Blue (Median)
        fig.add_trace(go.Scatter(
            x=results_df["Year_Float"][valid_p],
            y=results_df["P50_Median"][valid_p],
            mode="lines",
            line=dict(color="#29b6f6", width=3),
            name="Median Projection (P50)",
            customdata=hover_data_custom[valid_p],
            text=results_df["Year_Label"][valid_p],
            hovertemplate="%{text}: %{y:$.3s}<extra></extra>"
        ))

        # C. Red (Bear)
        fig.add_trace(go.Scatter(
            x=results_df["Year_Float"][valid_p],
            y=results_df["P10_Bear"][valid_p],
            mode="lines",
            line=dict(color="#ff5252", width=2, dash="dash"),
            name="10th Percentile (Bear)",
            customdata=hover_data_custom[valid_p],
            text=results_df["Year_Label"][valid_p],
            hovertemplate="%{text}: %{y:$.3s}<extra></extra>"
        ))

        # D. Recorded Actuals
        actual_mask = results_df["Actual"].notna()
        if actual_mask.any():
            fig.add_trace(go.Scatter(
                x=results_df["Year_Float"][actual_mask],
                y=results_df["Actual"][actual_mask],
                mode="lines+markers",
                line=dict(color="#ffffff", width=3),
                marker=dict(size=7, color="#ffffff", symbol="circle"),
                name="Actual Realized Net Worth",
                customdata=hover_data_custom[actual_mask],
                text=results_df["Year_Label"][actual_mask],
                hovertemplate="Actual %{text}: %{y:$.3s}<extra></extra>"
            ))

        # Dynamic Y-Axis Bounds
        all_min_candidates = []
        all_max_candidates = []

        if valid_p.any():
            all_min_candidates.append(results_df["P10_Bear"][valid_p].min())
            all_max_candidates.append(results_df["P90_Bull"][valid_p].max())

        if actual_mask.any():
            all_min_candidates.append(results_df["Actual"][actual_mask].min())
            all_max_candidates.append(results_df["Actual"][actual_mask].max())

        if all_min_candidates and all_max_candidates:
            y_min_val = min(all_min_candidates)
            y_max_val = max(all_max_candidates)

            y_spread = y_max_val - y_min_val
            y_lower_bound = max(0.0, y_min_val - (0.05 * y_spread))
            y_upper_bound = y_max_val + (0.05 * y_spread)
            y_range = [y_lower_bound, y_upper_bound]
        else:
            y_range = None

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            xaxis=dict(
                title="Time Horizon",
                dtick=1,
                tick0=0,
                range=[-0.5, total_horizon],
                autorange=False,
                gridcolor="#262730",
                zerolinecolor="#262730",
                minor=dict(
                    dtick=0.25,
                    showgrid=True,
                    gridcolor="#1e2029",
                    gridwidth=1
                )
            ),
            yaxis=dict(
                title="Net Worth",
                tickformat="$.3s",
                gridcolor="#262730",
                zerolinecolor="#262730",
                range=y_range,
                autorange=False if y_range else True
            ),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=11),
                traceorder="normal"
            ),
            hoverlabel=dict(
                namelength=-1
            ),
            margin=dict(l=0, r=0, t=30, b=0)
        )

        st.plotly_chart(fig, width='stretch')

    # --- DATA SUMMARY TABLE ---
    st.divider()
    st.subheader("Data Summary Table")

    formatted_df = results_df.copy()
    for col in ["Actual", "P10_Bear", "P50_Median", "P90_Bull"]:
        formatted_df[col] = formatted_df[col].apply(
            lambda x: f"${x / 1e6:.3f}M" if pd.notna(x) and x >= 1e6
            else (f"${x:,.0f}" if pd.notna(x) else "")
        )

    st.dataframe(formatted_df[["Year", "Quarter", "Actual", "P10_Bear", "P50_Median", "P90_Bull"]],
                 width='stretch')

except Exception as e:
    st.error(f"Error running simulation: {e}")