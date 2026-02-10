from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

_SRC_DIR = Path(__file__).parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from lp_index import build_lp_index_levels, compute_normalized_returns, load_constituents


APP_TITLE = "Labels & Packaging (L&P) vs S&P 500"
CONSTITUENTS_CSV = Path(__file__).parent / "data" / "constituents.csv"
CORRUGATED_CSV = Path(__file__).parent / "data" / "corrugated_constituents.csv"


RANGE_OPTIONS = {
    "1D": ("days", 2),
    "5D": ("days", 7),
    "1M": ("months", 1),
    "6M": ("months", 6),
    "YTD": ("ytd", None),
    "1Y": ("years", 1),
    "2Y": ("years", 2),
    "5Y": ("years", 5),
    "All": ("max", None),
}


def _start_date_for_range(range_key: str) -> datetime | None:
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    kind, value = RANGE_OPTIONS[range_key]

    if kind == "max":
        return None
    if kind == "ytd":
        return datetime(today.year, 1, 1)

    if kind == "days":
        return (today - pd.Timedelta(days=int(value))).to_pydatetime()
    if kind == "months":
        return (today - pd.DateOffset(months=int(value))).to_pydatetime()
    if kind == "years":
        return (today - pd.DateOffset(years=int(value))).to_pydatetime()

    return None


@st.cache_data(show_spinner=False)
def _fetch_prices(tickers: list[str], start: datetime | None):
    # auto_adjust=False gives both Close and Adj Close.
    df = yf.download(
        tickers,
        start=start,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    return df


@st.cache_data(show_spinner=False)
def _fetch_shares_outstanding(tickers: list[str]) -> pd.Series:
    shares = {}
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            val = None

            # Prefer fast_info when available.
            try:
                fi = getattr(tk, "fast_info", None)
                if fi and isinstance(fi, dict):
                    val = fi.get("shares") or fi.get("sharesOutstanding")
            except Exception:
                pass

            if not val:
                try:
                    info = tk.get_info()
                    if isinstance(info, dict):
                        val = info.get("sharesOutstanding")
                except Exception:
                    val = None

            shares[t] = float(val) if val else None
        except Exception:
            shares[t] = None

    return pd.Series(shares, dtype="float64", name="shares_outstanding")


def _extract_field(df: pd.DataFrame, tickers: list[str], field: str) -> pd.DataFrame:
    """Extract a single OHLCV field from yfinance download output."""

    if isinstance(df.columns, pd.MultiIndex):
        # MultiIndex: (field, ticker) OR (ticker, field) depending on yfinance.
        if df.columns.nlevels != 2:
            raise ValueError("Unexpected columns format from yfinance")

        # Try (field, ticker)
        if field in df.columns.get_level_values(0):
            out = df[field]
            return out.reindex(columns=tickers)

        # Try (ticker, field)
        if field in df.columns.get_level_values(1):
            cols = pd.MultiIndex.from_product([tickers, [field]])
            out = df.reindex(columns=cols)
            out.columns = [c[0] for c in out.columns]
            return out

        raise ValueError(f"Field '{field}' not present in downloaded data")

    # SingleIndex (one ticker)
    if field in df.columns:
        return df[[field]].rename(columns={field: tickers[0]})
    raise ValueError(f"Field '{field}' not present in downloaded data")


def _infer_currency(ticker: str) -> str:
    suffix_map = {
        ".TO": "CAD",
        ".L": "GBP",
        ".HE": "EUR",
        ".VI": "EUR",
        ".T": "JPY",
        ".SW": "CHF",
        ".DE": "EUR",
        ".PA": "EUR",
        ".MI": "EUR",
        ".HK": "HKD",
        ".AX": "AUD",
        ".ST": "SEK",
        ".OL": "NOK",
        ".SA": "BRL",
    }
    for suffix, ccy in suffix_map.items():
        if ticker.endswith(suffix):
            return ccy
    return "USD"


@st.cache_data(show_spinner=False)
def _fetch_fx_rates(currencies: list[str], start: datetime | None) -> pd.DataFrame:
    fx_ticker_by_ccy = {
        "EUR": "EURUSD=X",
        "JPY": "JPYUSD=X",
        "GBP": "GBPUSD=X",
        "CAD": "CADUSD=X",
        "CHF": "CHFUSD=X",
        "AUD": "AUDUSD=X",
        "HKD": "HKDUSD=X",
        "SEK": "SEKUSD=X",
        "NOK": "NOKUSD=X",
        "BRL": "BRLUSD=X",
    }

    fx_tickers = [fx_ticker_by_ccy[c] for c in currencies if c in fx_ticker_by_ccy]
    if not fx_tickers:
        return pd.DataFrame()

    raw = _fetch_prices(fx_tickers, start=start)
    fx_close = _extract_field(raw, fx_tickers, "Close")

    ccy_by_fx = {v: k for k, v in fx_ticker_by_ccy.items()}
    fx_close = fx_close.rename(columns=ccy_by_fx)
    return fx_close


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    shares_file = st.file_uploader(
        "Optional: Upload shares_outstanding CSV",
        type=["csv"],
        help="CSV with columns: ticker, shares_outstanding",
    )

    return_type = st.radio(
        "Return type",
        options=["Price return", "Total return (Adj Close)"],
        horizontal=True,
        index=1,
    )

    universe = st.radio(
        "Universe",
        options=["Indigo", "Corrugated"],
        horizontal=True,
        index=0,
    )

    reconstitution = st.selectbox(
        "Reconstitution frequency",
        options=["None (static list)", "Monthly", "Quarterly", "Annual"],
        index=2,
    )

    show_company = st.checkbox("Add single company line", value=False)

    range_key = st.radio(
        "Range",
        options=list(RANGE_OPTIONS.keys()),
        index=list(RANGE_OPTIONS.keys()).index("1Y"),
        horizontal=True,
    )

    base_constituents = load_constituents(CONSTITUENTS_CSV)
    if universe == "Corrugated":
        constituents = load_constituents(CORRUGATED_CSV)
        constituents_source = "data/corrugated_constituents.csv"

        overlap = constituents.index.intersection(base_constituents.index)
        if not overlap.empty:
            constituents.loc[
                overlap,
                ["shares_outstanding", "free_float", "capping_factor"],
            ] = base_constituents.loc[
                overlap,
                ["shares_outstanding", "free_float", "capping_factor"],
            ]
    else:
        constituents = base_constituents
        constituents_source = "data/constituents.csv"
    lp_tickers = constituents.index.tolist()
    name_by_ticker = constituents["name"].to_dict()
    display_name_by_ticker = {
        t: f"{t} ({name_by_ticker.get(t)})" if name_by_ticker.get(t) else t
        for t in lp_tickers
    }

    if shares_file is not None:
        try:
            shares_df = pd.read_csv(shares_file)
            if "ticker" in shares_df.columns and "shares_outstanding" in shares_df.columns:
                shares_df["ticker"] = (
                    shares_df["ticker"].astype(str).str.strip().str.upper()
                )
                shares_df["shares_outstanding"] = pd.to_numeric(
                    shares_df["shares_outstanding"], errors="coerce"
                )
                overrides = shares_df.dropna(subset=["ticker"]).set_index("ticker")
                constituents.loc[
                    constituents.index.intersection(overrides.index),
                    "shares_outstanding",
                ] = overrides.loc[
                    constituents.index.intersection(overrides.index),
                    "shares_outstanding",
                ]
            else:
                st.error("Uploaded CSV must include 'ticker' and 'shares_outstanding'.")
        except Exception:
            st.error("Could not read the uploaded CSV.")

    company_ticker = None
    if show_company:
        company_ticker = st.selectbox(
            "Company",
            options=lp_tickers,
            index=0,
            format_func=lambda t: display_name_by_ticker.get(t, t),
        )

    if len(lp_tickers) < 2:
        st.error(f"Add at least 2 tickers to {constituents_source}")
        st.stop()

    start = _start_date_for_range(range_key)

    use_total_return = return_type == "Total return (Adj Close)"
    sp_tickers = ["^GSPC"]
    if use_total_return:
        sp_tickers = ["^SP500TR", "^GSPC"]

    with st.spinner("Downloading market data..."):
        prices_raw = _fetch_prices(sp_tickers + lp_tickers, start=start)

    price_field = "Adj Close" if use_total_return else "Close"

    sp_label = "S&P 500"
    if use_total_return:
        sp_label = "S&P 500 TR"

    sp_series = None
    if use_total_return:
        try:
            sp_tr = _extract_field(prices_raw, ["^SP500TR"], price_field)["^SP500TR"]
            if sp_tr.notna().any():
                sp_series = sp_tr.rename(sp_label)
        except Exception:
            sp_series = None

    if sp_series is None:
        if use_total_return:
            st.warning("S&P 500 TR data not available; falling back to ^GSPC.")
        sp_series = _extract_field(prices_raw, ["^GSPC"], price_field)["^GSPC"].rename(
            "S&P 500"
        )

    lp_close = _extract_field(prices_raw, lp_tickers, price_field)
    # Clean and align: some tickers trade on different calendars.
    # Treat non-positive prices as missing and forward-fill short gaps.
    lp_close = lp_close.where(lp_close > 0)
    lp_close = lp_close.sort_index().ffill(limit=7)

    currency_by_ticker = {t: _infer_currency(t) for t in lp_tickers}
    fx_currencies = sorted({c for c in currency_by_ticker.values() if c != "USD"})
    fx_rates = _fetch_fx_rates(fx_currencies, start=start)
    if not fx_rates.empty:
        fx_rates = fx_rates.reindex(lp_close.index).sort_index().ffill(limit=7)

    missing_fx = []
    lp_close_usd = lp_close.copy()
    for ticker, ccy in currency_by_ticker.items():
        if ccy == "USD":
            continue
        if ccy not in fx_rates.columns or fx_rates[ccy].dropna().empty:
            missing_fx.append(ccy)
            continue
        
        # Yahoo Finance returns LSE (.L) prices in pence (GBp), but FX is in GBP.
        factor = 1.0
        if ticker.endswith(".L"):
            factor = 0.01

        lp_close_usd[ticker] = lp_close_usd[ticker] * factor * fx_rates[ccy]

    if missing_fx:
        st.warning(
            "Missing FX rates for: "
            + ", ".join(sorted(set(missing_fx)))
            + ". Prices for those tickers were left in local currency."
        )

    lp_close = lp_close_usd

    with st.spinner("Fetching shares outstanding (public data)..."):
        shares = constituents["shares_outstanding"].copy()
        missing = shares[shares.isna()].index.tolist()
        if missing:
            fetched = _fetch_shares_outstanding(missing)
            shares.loc[missing] = fetched

    usable = shares.dropna().index.tolist()
    dropped = sorted(set(lp_tickers) - set(usable))
    if dropped:
        dropped_labels = [display_name_by_ticker.get(t, t) for t in dropped]
        st.warning(
            "Dropped tickers with missing shares_outstanding: "
            + ", ".join(dropped_labels)
        )

    if len(usable) < 2:
        st.error(
            "Not enough shares_outstanding data to build a market-cap-weighted index. "
            "Please fill shares_outstanding in data/constituents.csv or ensure Yahoo data is available."
        )
        st.stop()

    lp_levels = build_lp_index_levels(
        price_frame=lp_close.reindex(columns=usable),
        shares_outstanding=shares,
        free_float=constituents["free_float"],
        capping_factor=constituents["capping_factor"],
        base_value=1000.0,
    )

    sp_ret = compute_normalized_returns(sp_series)
    lp_ret = compute_normalized_returns(lp_levels)

    company_ret = None
    if company_ticker:
        company_series = lp_close[company_ticker]
        company_ret = compute_normalized_returns(company_series).rename(company_ticker)

    # Align on common dates
    series_list = [lp_ret.rename("L&P"), sp_ret.rename(sp_series.name)]
    if company_ret is not None:
        series_list.append(company_ret)

    df = pd.concat(series_list, axis=1)
    df = df.dropna(subset=["L&P", sp_series.name])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["L&P"] * 100.0,
            mode="lines",
            name="Labels & Packaging (L&P)",
            line=dict(color="#1f77b4"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[sp_series.name] * 100.0,
            mode="lines",
            name=sp_series.name,
            line=dict(color="#ff7f0e"),
        )
    )

    if company_ret is not None:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[company_ticker] * 100.0,
                mode="lines",
                name=company_ticker,
                line=dict(color="#d62728"),
            )
        )

    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis_title="% change",
        xaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption("Non-USD tickers are converted to USD using daily FX rates.")

    if reconstitution != "None (static list)":
        st.info(
            "Reconstitution is selected but the current constituents list is static. "
            "When you provide a time series of top-50 members, the app will apply this schedule."
        )


if __name__ == "__main__":
    main()
