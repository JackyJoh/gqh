"""Backtest: the locked rules, the price matrices, and portfolio returns.

The locked strategy: every session, hold the stocks ranked above the 95th percentile (top 5%, ~25 stocks), equal
weight, bought at the closing auction and sold at the next opening auction. Each night's return is booked on the
session of the sell (the open). Costs are a flat deduction per side; a night has two sides (buy + sell).
"""
import numpy as np
import pandas as pd
from data import download as dl
from . import signals

DATA_START = "2008-12-01"     # 2009 = warm-up, so the first ranking (2009-12-31) has a full 252-session window
TRADE_START = "2010-01-01"    # first session with a position: 2010-01-04
IS_END = "2024-10-02"         # in-sample: 2010-01-04 .. 2024-10-02
OOS_START, OOS_END = "2024-10-03", "2026-10-02"   # out-of-sample (run once, rules locked)
OOS_DATA_START = "2023-09-01" # the first OOS ranking (2024-09-30) needs 252 prior sessions; this covers it with margin
CUTOFF = 95                   # top 5%
COST = 0.0002                 # 2 bps per side (headline); 4 bps per side = costs doubled
SAMPLE_FROM, SAMPLE_EVERY = "2015-01-01", 5   # execution check: every 5th session from 2015


def load_bars(start, end, log=print):
    """Open / close / volume matrices (sessions x tickers) for start..end, downloading any missing days first.

    Columns = US common stocks with at least one day of >= $20M dollar volume at price >= $5 inside start..end
    (a speed-up: anything that never reaches that can't be in the top-500 universe). SPY (open, close) is returned
    separately as the benchmark. Only data inside start..end is used, so an in-sample run never sees later data."""
    cs = set(dl.tickers().ticker)
    dl.daily_bars(start, end, log=log)
    frames, spy, liquid = [], {}, set()
    for d in pd.bdate_range(start, end).strftime("%Y-%m-%d"):
        p = pd.read_parquet(dl.OCV / f"{d}.parquet")
        if len(p):
            s = p[p.ticker == "SPY"]
            if len(s):
                spy[pd.Timestamp(d)] = (float(s.open.iloc[0]), float(s.close.iloc[0]))
            p = p[p.ticker.isin(cs)]
            liquid |= set(p.ticker[(p.close * p.volume >= 20e6) & (p.close >= 5)])
            frames.append(p.assign(date=pd.Timestamp(d)))
    P = pd.concat(frames)
    P = P[P.ticker.isin(liquid)].drop_duplicates(["date", "ticker"])
    O = P.pivot(index="date", columns="ticker", values="open").sort_index()
    C = P.pivot(index="date", columns="ticker", values="close").reindex_like(O)
    V = P.pivot(index="date", columns="ticker", values="volume").reindex_like(O)
    spy = pd.DataFrame.from_dict(spy, orient="index", columns=["open", "close"]).reindex(O.index)
    log(f"   {len(O):,} sessions x {O.shape[1]:,} stocks ({O.index[0].date()} .. {O.index[-1].date()})")
    return O, C, V, spy


def build(start, end, log=print):
    """Everything the strategy needs, using only data from start..end."""
    O, C, V, spy = load_bars(start, end, log=log)
    ON = signals.overnight_returns(O, C)
    top = signals.universe(C, V)
    sc = signals.score(ON)
    pct = signals.monthly_percentiles(sc, top)
    return {"O": O, "C": C, "V": V, "ON": ON, "top": top, "score": sc, "pct": pct, "S": O.index,
            "spy": spy.close.pct_change(fill_method=None),                 # SPY buy & hold (close to close)
            "spy_on": spy.open / spy.close.shift(1) - 1}                    # SPY overnight (close to next open)


def portfolio(ON, pct, lo=CUTOFF, hi=100):
    """Gross daily return of an equal-weight book holding stocks with lo < percentile <= hi, and stocks held per day.
    A day with nothing held returns 0; a held stock with no next open (halt / delisting) contributes nothing."""
    held = (pct > lo) & (pct <= hi)
    return ON.where(held).mean(axis=1).fillna(0), held.sum(axis=1)


def net(gross, cost_side=COST):
    """Daily return after paying cost_side on the buy and on the sell."""
    return gross - 2 * cost_side


def sharpe(r):
    return r.mean() / r.std() * np.sqrt(252)


def walk_forward(cut_gross, first_year=2012):
    """Each calendar year, trade the cutoff with the best net Sharpe over all PRIOR years only (no hindsight).
    cut_gross = {name: gross daily returns}. Returns (daily gross returns from first_year, {year: pick})."""
    idx = next(iter(cut_gross.values())).index
    wf, picks = pd.Series(0.0, index=idx), {}
    for y in sorted(set(idx.year)):
        if y < first_year:
            continue
        past = {k: sharpe(net(v[v.index.year < y])) for k, v in cut_gross.items()}
        best = max(past, key=past.get)
        picks[y] = best
        wf[idx.year == y] = cut_gross[best][idx.year == y]
    return wf[wf.index.year >= first_year], picks


def robustness(ON, top, start="2011-07-01", end=IS_END, log=print):
    """Top 5% for ranking windows of 1/3/6/12/24 months x daily/weekly/monthly re-ranking, over a common period
    (the 24-month window needs history). Rank at the close of a rebalance date; hold until the next one.
    Returns {(window, rerank): gross daily returns}, {(window, rerank): stocks held per day}."""
    S = ON.index
    period = (S >= start) & (S <= end)
    wk = pd.Series(S, index=S)
    rebal = {"daily": S,
             "weekly": pd.DatetimeIndex(wk.groupby([S.isocalendar().year.values, S.isocalendar().week.values]).max().values),
             "monthly": pd.DatetimeIndex(wk.groupby([S.year, S.month]).max().values)}
    out, n = {}, {}
    for wname, w in {"1m": 21, "3m": 63, "6m": 126, "12m": 252, "24m": 504}.items():
        sc = ON.rolling(w, min_periods=int(0.8 * w)).mean().where(top)
        pick = (sc.rank(axis=1, pct=True) > 0.95) & (sc.notna().sum(axis=1) >= signals.MIN_RANKED).values[:, None]
        for fname, dates in rebal.items():
            h = pick.loc[pick.index.isin(dates)].reindex(S).ffill().shift(1).fillna(False).astype(bool)
            out[wname, fname] = ON.where(h).mean(axis=1).fillna(0)[period]
            n[wname, fname] = h[period].sum(axis=1)
        log(f"   window {wname} done")
    return out, n


def sample_positions(b, end):
    """Execution-check sample: top-DECILE positions (a superset of the top 5%, so each can be tagged top 1/2/3/5/10%)
    on every SAMPLE_EVERY-th session from SAMPLE_FROM to end. b = build() output."""
    S, C, O, sc, top = b["S"], b["C"], b["O"], b["score"], b["top"]
    nights = set(S[(S >= SAMPLE_FROM) & (S <= end)][::SAMPLE_EVERY])
    me = signals.month_ends(S)
    rows = []
    for i, d in enumerate(me[:-1]):
        s = sc.loc[d].where(top.loc[d]).dropna()
        if len(s) < signals.MIN_RANKED:
            continue
        q = pd.qcut(s, 10, labels=False) + 1
        names = q.index[q == 10]
        for t in S[(S > d) & (S <= me[i + 1])]:
            if t in nights:
                prev = S[S.get_loc(t) - 1]
                rows += [(t, prev, tk, C.at[prev, tk], O.at[t, tk]) for tk in names]
    return pd.DataFrame(rows, columns=["date", "prev", "ticker", "close_print", "open_print"])
