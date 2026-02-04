from pathlib import Path

import pandas as pd
import yfinance as yf

SRC = Path(r"c:\Index\data\shares_outstanding_upload.csv")
if not SRC.exists():
    raise FileNotFoundError(SRC)

shares_df = pd.read_csv(SRC)
if "ticker" not in shares_df.columns:
    raise ValueError("CSV missing ticker column")

shares_df["ticker"] = shares_df["ticker"].astype(str).str.strip().str.upper()


def fetch_shares(ticker: str):
    try:
        tk = yf.Ticker(ticker)
        val = None
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
        return float(val) if val else None
    except Exception:
        return None


filled = []
for t in shares_df["ticker"].tolist():
    if t:
        filled.append(fetch_shares(t))
    else:
        filled.append(None)

shares_df["shares_outstanding"] = filled
shares_df.to_csv(SRC, index=False)
print(SRC)
