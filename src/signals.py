"""Signal: overnight returns, the point-in-time universe, the trailing overnight score and month-end percentile ranks."""
import numpy as np
import pandas as pd

MASK = 0.5            # |overnight return| > 50% is treated as a bad print / split artifact and dropped
UNIVERSE_N = 500      # top 500 by 20-day median dollar volume ...
MIN_PRICE = 5.0       # ... with price >= $5
WINDOW = 252          # score = mean overnight return over the last 252 sessions (~1 year) ...
MIN_DAYS = 200        # ... with at least 200 valid days
MIN_RANKED = 300      # skip a month (stay flat) if fewer stocks than this have a score


def overnight_returns(O, C):
    """Overnight return on day t = open_t / close_{t-1} - 1 (buy at the close, sell at the next open)."""
    ON = O / C.shift(1) - 1
    return ON.mask(ON.abs() > MASK)


def universe(C, V):
    """True where a stock is in the top 500 by 20-day median dollar volume (close x volume) with price >= $5.
    Uses data up to and including that day's close."""
    dvol = (C * V).rolling(20, min_periods=10).median()
    return dvol.where(C >= MIN_PRICE).rank(axis=1, ascending=False) <= UNIVERSE_N


def score(ON):
    """Trailing mean overnight return over the last WINDOW sessions."""
    return ON.rolling(WINDOW, min_periods=MIN_DAYS).mean()


def month_ends(S):
    """Last session of each calendar month in the session index S."""
    return pd.Series(S, index=S).groupby([S.year, S.month]).max().values


def monthly_percentiles(sc, top):
    """Rank the universe by score at each month-end close (percentile 0-100) and hold that rank for every session of
    the NEXT month. NaN = not ranked (outside the universe, no score, or a skipped month)."""
    S = sc.index
    me = month_ends(S)
    pct = pd.DataFrame(np.nan, index=S, columns=sc.columns)
    for i, d in enumerate(me[:-1]):
        s = sc.loc[d].where(top.loc[d]).dropna()
        if len(s) < MIN_RANKED:
            continue
        pct.loc[(S > d) & (S <= me[i + 1]), s.index] = (s.rank(pct=True) * 100).values
    return pct
