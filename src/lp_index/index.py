from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd


IndexMode = Literal["price_return", "total_return"]


@dataclass(frozen=True)
class Constituent:
    ticker: str
    name: str | None = None
    shares_outstanding: float | None = None
    free_float: float = 1.0
    capping_factor: float = 1.0


def load_constituents(csv_path: str | Path) -> pd.DataFrame:
    """Load constituents CSV.

    Expected columns:
      - ticker (required)
      - name (optional)
      - shares_outstanding (optional)
      - free_float (optional, default 1.0)
      - capping_factor (optional, default 1.0)

    Returns a cleaned DataFrame indexed by ticker.
    """

    df = pd.read_csv(csv_path)
    if "ticker" not in df.columns:
        raise ValueError("Constituents file must contain a 'ticker' column")

    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df[df["ticker"].ne("")]

    if "name" not in df.columns:
        df["name"] = np.nan

    for col, default in ("free_float", 1.0), ("capping_factor", 1.0):
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    if "shares_outstanding" not in df.columns:
        df["shares_outstanding"] = np.nan
    df["shares_outstanding"] = pd.to_numeric(df["shares_outstanding"], errors="coerce")

    df = df.drop_duplicates(subset=["ticker"], keep="first")
    df = df.set_index("ticker", drop=True)
    return df


def compute_normalized_returns(levels: pd.Series) -> pd.Series:
    levels = levels.dropna()
    if levels.empty:
        return levels
    return levels / levels.iloc[0] - 1.0


def build_lp_index_levels(
    price_frame: pd.DataFrame,
    shares_outstanding: pd.Series,
    free_float: pd.Series | None = None,
    capping_factor: pd.Series | None = None,
    base_value: float = 1000.0,
) -> pd.Series:
    """Build an S&P-style float-adjusted market-cap weighted index level series.

    Core formula (static basket, no divisor adjustments needed):

        I_t = Sum_i(P_{i,t} * Q_i * FF_i * CF_i) / D

    Where divisor D is chosen so I_0 = base_value.

    Notes:
    - This implementation assumes a fixed constituent set for the whole period.
    - For real S&P-like maintenance (adds/deletes, float updates, corporate actions),
      the divisor would be adjusted on event dates.
    """

    if price_frame.empty:
        return pd.Series(dtype=float)

    tickers = [c for c in price_frame.columns]
    shares = shares_outstanding.reindex(tickers).astype(float)

    ff = (
        free_float.reindex(tickers).astype(float)
        if free_float is not None
        else pd.Series(1.0, index=tickers, dtype=float)
    )
    cf = (
        capping_factor.reindex(tickers).astype(float)
        if capping_factor is not None
        else pd.Series(1.0, index=tickers, dtype=float)
    )

    usable = shares.notna() & np.isfinite(shares)
    tickers_usable = shares.index[usable].tolist()
    if len(tickers_usable) == 0:
        raise ValueError(
            "No usable shares_outstanding values were provided. "
            "Fill 'shares_outstanding' in constituents.csv or enable fetching in the app."
        )

    prices = price_frame[tickers_usable].astype(float)
    shares = shares.loc[tickers_usable]
    ff = ff.loc[tickers_usable]
    cf = cf.loc[tickers_usable]

    float_adj_mcap = prices.mul(shares, axis=1).mul(ff, axis=1).mul(cf, axis=1)
    total_mcap = float_adj_mcap.sum(axis=1, min_count=1)

    total_mcap = total_mcap.dropna()
    if total_mcap.empty:
        return pd.Series(dtype=float)

    divisor = total_mcap.iloc[0] / float(base_value)
    if divisor == 0:
        raise ValueError("Divisor computed as 0; check inputs")

    return (total_mcap / divisor).rename("L&P")
