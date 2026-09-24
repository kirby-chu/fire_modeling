"""
engine.py - Quarterly Monte Carlo FIRE Engine (Strict Type Safety)
"""
from dataclasses import dataclass, field, asdict, fields
from typing import Dict, Optional, Tuple
import json
import os
import numpy as np
import pandas as pd

DEFAULT_FILE_PATH = "fire_profile.json"


@dataclass
class FIRERunner:
    salary_1: float
    salary_2: float
    annual_savings_amount: Optional[float] = None
    savings_rate: Optional[float] = None
    income_growth_rate: float = 0.03  # Annual wage growth
    mean_return: float = 0.07  # Annual real return post-inflation
    volatility: float = 0.16  # Annualized standard deviation
    num_simulations: int = 5_000  # Number of Monte Carlo trajectories
    seed: int = 42  # Random seed for reproducibility

    # Track actual net worth check-ins indexed by 0-based quarter steps internally: {q_step: value}
    # Step 0 = Y1 Q0 (Baseline Start)
    actuals: Dict[int, float] = field(default_factory=dict)

    def run_simulation(
            self,
            total_horizon_years: int = 20,
            num_simulations: Optional[int] = None,
            seed: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Runs Monte Carlo simulation on a quarterly basis using Geometric Brownian Motion.
        Supports both positive savings (accumulation) and negative amounts (decumulation/retirement).
        """
        n_sims = int(num_simulations if num_simulations is not None else self.num_simulations)
        rnd_seed = int(seed if seed is not None else self.seed)

        np.random.seed(rnd_seed)

        total_quarters = int(total_horizon_years * 4)

        start_step = max(self.actuals.keys()) if self.actuals else 0
        starting_capital = float(self.actuals.get(start_step, 0.0))

        remaining_steps = total_quarters - start_step
        if remaining_steps < 0:
            raise ValueError("Recorded actuals extend past the total horizon.")

        # Convert annual parameters to quarterly equivalents
        q_mean_return = float(np.power(1.0 + self.mean_return, 0.25) - 1.0)
        q_volatility = float(self.volatility / np.sqrt(4.0))
        q_wage_growth = float(np.power(1.0 + self.income_growth_rate, 0.25) - 1.0)

        total_gross = float(self.salary_1 + self.salary_2)

        sim_paths = np.zeros((n_sims, remaining_steps + 1), dtype=np.float64)
        sim_paths[:, 0] = starting_capital

        for step in range(1, remaining_steps + 1):
            q_curr = start_step + step

            # Contribution Mode Logic (Supports positive contributions or negative withdrawals)
            if self.savings_rate is not None:
                current_q_salary = total_gross * ((1.0 + q_wage_growth) ** (q_curr - 1))
                contribution = (current_q_salary * float(self.savings_rate)) / 4.0
            else:
                contribution = float(self.annual_savings_amount) / 4.0

            if q_volatility > 0:
                shock = np.random.normal(0, 1, n_sims)
                returns = np.exp((q_mean_return - 0.5 * (q_volatility ** 2)) + q_volatility * shock)
            else:
                returns = 1.0 + q_mean_return

            # Compound returns and add net contribution/withdrawal
            next_state = (sim_paths[:, step - 1] * returns) + contribution

            # Floor portfolio at $0 to correctly represent bankruptcy rather than unconstrained debt
            sim_paths[:, step] = np.maximum(0.0, next_state)

        # Build results Summary DataFrame with EXPLICIT FLOAT DTYPES to prevent casting errors
        q_steps = np.arange(total_quarters + 1, dtype=np.int64)

        years = [int(1 if step == 0 else ((step - 1) // 4) + 1) for step in q_steps]
        quarters = [int(0 if step == 0 else ((step - 1) % 4) + 1) for step in q_steps]
        year_floats = (q_steps / 4.0).astype(np.float64)
        year_labels = [f"Y{y} Q{q}" for y, q in zip(years, quarters)]

        # Calculate Percentiles as float64
        p10_future = np.percentile(sim_paths, 10, axis=0).astype(np.float64)
        p50_future = np.percentile(sim_paths, 50, axis=0).astype(np.float64)
        p90_future = np.percentile(sim_paths, 90, axis=0).astype(np.float64)

        actual_series = np.full(len(q_steps), np.nan, dtype=np.float64)
        p10_series = np.full(len(q_steps), np.nan, dtype=np.float64)
        p50_series = np.full(len(q_steps), np.nan, dtype=np.float64)
        p90_series = np.full(len(q_steps), np.nan, dtype=np.float64)

        for step_idx in range(len(q_steps)):
            if step_idx in self.actuals:
                actual_series[step_idx] = float(self.actuals[step_idx])

        p10_series[start_step:] = p10_future
        p50_series[start_step:] = p50_future
        p90_series[start_step:] = p90_future

        # Patch pre-anchor historical steps
        for q in range(start_step):
            if q in self.actuals:
                val = float(self.actuals[q])
                p10_series[q] = val
                p50_series[q] = val
                p90_series[q] = val

        df_summary = pd.DataFrame({
            "Year": years,
            "Quarter": quarters,
            "Year_Float": year_floats,
            "Year_Label": year_labels,
            "Actual": actual_series,
            "P10_Bear": p10_series,
            "P50_Median": p50_series,
            "P90_Bull": p90_series
        }, index=q_steps)
        df_summary.index.name = "Step"

        sample_size = min(50, n_sims)
        sample_paths_data = {}

        for i in range(sample_size):
            path_col = np.full(len(q_steps), np.nan, dtype=np.float64)
            path_col[start_step:] = sim_paths[i, :]
            sample_paths_data[f"Run_{i}"] = path_col

        df_paths = pd.DataFrame(sample_paths_data, index=q_steps)
        df_paths["Year_Float"] = year_floats
        df_paths.index.name = "Step"

        return df_summary, df_paths

    def to_json(self) -> str:
        """Serializes parameter state and actuals dictionary to JSON string."""
        data = asdict(self)
        data["actuals"] = {str(k): float(v) for k, v in self.actuals.items()}
        return json.dumps(data, indent=2)

    def save_to_file(self, filepath: str = DEFAULT_FILE_PATH):
        """Saves current state to local JSON file."""
        with open(filepath, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_json(cls, json_str: str) -> "FIRERunner":
        """Reconstructs FIRERunner instance from JSON string, safely filtering stale keys."""
        data = json.loads(json_str)
        actuals_raw = data.pop("actuals", {})
        actuals_parsed = {int(k): float(v) for k, v in actuals_raw.items()}

        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        return cls(**filtered_data, actuals=actuals_parsed)

    @classmethod
    def load_from_file(cls, filepath: str = DEFAULT_FILE_PATH) -> Optional["FIRERunner"]:
        """Loads FIRERunner instance from local file if it exists."""
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                return cls.from_json(f.read())
        return None