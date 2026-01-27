# Labels & Packaging (L&P) vs S&P 500

This app charts the S&P 500 index (`^GSPC`) versus a custom index called **L&P** (Labels & Packaging), built from a list of public companies.

## What you get

- A 1Y-style chart (with range buttons) showing **% change** for:
  - **S&P 500** (Yahoo Finance ticker `^GSPC`)
  - **L&P** (your custom basket)

## How the L&P index is calculated (S&P-style)

The S&P 500 is a **float-adjusted market-cap weighted index** maintained using a **divisor** so index changes reflect market value changes, not mechanical effects.

This project implements the same *core* mechanism for a **fixed constituent set**:

### 1) Float-adjusted market cap per stock

For each constituent $i$ on date $t$:

$$\text{FloatAdjMCap}_{i,t} = P_{i,t} \times Q_i \times FF_i \times CF_i$$

Where:
- $P_{i,t}$ = stock price (we use Yahoo Finance **Close** for price return or **Adj Close** for total return)
- $Q_i$ = shares outstanding (from Yahoo Finance company info; or you can provide it)
- $FF_i$ = free-float factor (defaults to 1.0)
- $CF_i$ = capping factor (defaults to 1.0)

### 2) Total float-adjusted market cap

$$\text{TotalMCap}_t = \sum_i \text{FloatAdjMCap}_{i,t}$$

### 3) Divisor and index level

We pick a base value (default 1000) and set the divisor using the first date in the chart window:

$$D = \frac{\text{TotalMCap}_{t_0}}{\text{Base}}$$

Then the index level is:

$$I_t = \frac{\text{TotalMCap}_t}{D}$$

### What this does *not* implement (yet)

Real S&P indices adjust the divisor for **adds/deletes, float share updates, corporate actions**, etc. If you keep a **static list** of companies over the chart window, no divisor adjustments are required beyond setting the initial base.

The app includes a **reconstitution frequency selector** (Monthly/Quarterly/Annual), but with a static list it does not change results yet. Once you provide a historical top-50 membership time series, the schedule can be applied to change constituents on those dates.

## Data you need to finalize (to match your exact definition)

To build your “top 50 Labels & Packaging” basket precisely, please confirm:

1) **Final list of 50 tickers** (and their primary listing). Some names you mentioned (e.g. CCL) may be on non-US exchanges or OTC symbols.
2) Do you want **price return** (like the common headline S&P 500 index) or **total return**? The app now has a toggle:
  - **Price return** uses `Close` for all series.
  - **Total return** uses `Adj Close` for L&P and tries `^SP500TR` for the S&P 500 (falls back to `^GSPC` if unavailable).
3) Rebalancing rules:
   - “Top 50” as of what date?
   - How often do you reconstitute/rebalance (monthly/quarterly/annual)?

## Configure constituents

Edit: `data/constituents.csv`

Columns:
- `ticker` (required)
- `name` (optional)
- `shares_outstanding` (optional; app will try to fetch it if blank)
- `free_float` (optional, default 1.0)
- `capping_factor` (optional, default 1.0)

## Run

1) Install deps:

```powershell
python -m pip install -r requirements.txt
```

2) Start the app:

```powershell
streamlit run app.py
```
