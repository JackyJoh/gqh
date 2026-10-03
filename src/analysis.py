"""Analysis: performance statistics, the execution-check and decile tables, and charts.

Sharpe = annualized mean daily return / annualized volatility, no risk-free rate subtracted (252 sessions per year).
"""
import numpy as np
import pandas as pd


def stats(r, n_held=None):
    """Full statistics of a daily return series (already net of costs)."""
    eq = (1 + r).cumprod()
    m = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    y = (1 + r).groupby(r.index.year).prod() - 1
    down = r[r < 0].std() * np.sqrt(252)
    out = {"$100k ->": eq.iloc[-1] * 1e5, "total %": (eq.iloc[-1] - 1) * 100,
           "CAGR %": (eq.iloc[-1] ** (252 / len(r)) - 1) * 100, "vol %": r.std() * np.sqrt(252) * 100,
           "Sharpe": r.mean() / r.std() * np.sqrt(252), "Sortino": r.mean() * 252 / down if down else np.nan,
           "max DD %": (eq / eq.cummax() - 1).min() * 100, "worst day %": r.min() * 100,
           "worst month %": m.min() * 100, "best month %": m.max() * 100, "positive months %": (m > 0).mean() * 100,
           "positive years %": (y > 0).mean() * 100, "sessions": len(r)}
    if n_held is not None:
        out["avg stocks"] = float(n_held[n_held > 0].mean())
    return out


TABLE1_COLS = ["$100k ->", "total %", "CAGR %", "vol %", "Sharpe", "Sortino", "max DD %", "worst day %",
               "worst month %", "best month %", "positive months %"]      # same columns for in-sample and OOS


def pct(v, d=1, sign=False):
    """Percent text from a fraction; never prints -0.0."""
    s = f"{v * 100:{'+' if sign else ''}.{d}f}"
    return f"{0:.{d}f}" if float(s) == 0 else s


def table1_text(rows):
    """rows = {name: stats(...)} -> aligned text with the shared Table 1 columns."""
    import pandas as pd
    out = {}
    for name, s in rows.items():
        out[name] = {"$100k ->": f"${s['$100k ->']:,.2f}", "total %": f"{s['total %']:+.1f}%", "CAGR %": f"{s['CAGR %']:.1f}%",
                     "vol %": f"{s['vol %']:.1f}%", "Sharpe": f"{s['Sharpe']:.2f}", "Sortino": f"{s['Sortino']:.2f}",
                     "max DD %": f"{s['max DD %']:.1f}%", "worst day %": f"{s['worst day %']:.1f}%",
                     "worst month %": f"{s['worst month %']:.1f}%", "best month %": f"{s['best month %']:+.1f}%",
                     "positive months %": f"{s['positive months %']:.0f}%"}
    t = pd.DataFrame(out).T[TABLE1_COLS]
    t.columns = [c.replace(" %", "") for c in t.columns]
    return t.to_string()


def sharpe(r):
    return r.mean() / r.std() * np.sqrt(252)


def by_year(r):
    return (1 + r).groupby(r.index.year).prod() - 1


def by_month(r):
    return (1 + r).groupby([r.index.year, r.index.month]).prod() - 1


def money(v):
    return f"${v / 1e6:,.2f}M" if v >= 1e6 else f"${v / 1e3:,.0f}k"


def growth_chart(lines, title, subtitle, out, log_scale=False, year_ticks=None):
    """Growth of $100k. lines = [(legend name, end label, daily returns, color), ...]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mt
    import pandas as pd

    SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for name, label, r, color in lines:
        eq = 1e5 * (1 + r).cumprod()
        ax.plot(eq.index, eq.values, color=color, lw=2, label=name)
        ax.annotate(f"{label}  {money(eq.iloc[-1])}", (eq.index[-1], eq.iloc[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.axhline(1e5, color=INK2, lw=1, ls=(0, (2, 3)))
    if log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mt.FixedLocator([1e5, 2e5, 5e5, 1e6, 2e6, 5e6]))
        ax.yaxis.set_minor_locator(mt.NullLocator())
    else:
        ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: money(v).replace(".00M", "M")))
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    first, last = lines[0][2].index[0], lines[0][2].index[-1]
    ax.set_xlim(first, last + (last - first) * 0.28)                 # room for the end labels
    if year_ticks:
        ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in year_ticks])
        ax.set_xticklabels([str(y) for y in year_ticks])
    else:                                                            # quarterly ticks, none past the last date
        import matplotlib.dates as md
        ax.set_xticks([t for t in pd.date_range(first, last, freq="QS")])
        ax.xaxis.set_major_formatter(md.DateFormatter("%b %Y"))
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=14, fontweight="bold")
    ax.text(0, 1.01, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK2)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def execution_table(px, pct, cost=0.0002):
    """Execution check per cutoff group, from sampled positions with auction prices and quotes (data/download.py):
    daily-bar vs official-auction return per night, break-even cost, auction Sharpe, median spreads near the close
    and at the open, and the return from buying the ask / selling the bid instead of using the auctions."""
    px = px.assign(pctile=pct.stack().reindex(pd.MultiIndex.from_arrays([px.date, px.ticker])).values)
    au = px.dropna(subset=["auc_close", "auc_open", "close_print", "open_print"])
    au = au[(au.auc_close > 0) & (au.auc_open > 0)]
    au = au.assign(r_print=au.open_print / au.close_print - 1, r_auction=au.auc_open / au.auc_close - 1)
    au = au[(au.r_auction - au.r_print).abs() <= 0.2]                    # drop split artifacts
    qt = px.dropna(subset=["bid_c", "ask_c", "bid_o", "ask_o"])
    mid_c, mid_o = (qt.bid_c + qt.ask_c) / 2, (qt.bid_o + qt.ask_o) / 2
    qt = qt.assign(spread_close=(qt.ask_c - qt.bid_c) / mid_c * 1e4, spread_open=(qt.ask_o - qt.bid_o) / mid_o * 1e4,
                   r_ask_bid=qt.bid_o / qt.ask_c - 1)
    rows = {}
    for name, lo in {"top 1%": 99, "top 2%": 98, "top 3%": 97, "top 5%": 95, "top 10%": 90}.items():
        a, k = au[au.pctile > lo], qt[qt.pctile > lo]
        na = a.groupby("date")[["r_print", "r_auction"]].mean()          # equal-weight book each sampled night
        x = na.r_auction - 2 * cost
        rows[name] = {"positions": float(len(a)), "daily-bar bps/night": na.r_print.mean() * 1e4,
                      "auction bps/night": na.r_auction.mean() * 1e4,
                      "break-even bps/side": na.r_auction.mean() / 2 * 1e4,
                      "auction Sharpe @2bp": x.mean() / x.std() * np.sqrt(252),
                      "close spread bps": k.spread_close.median(), "open spread bps": k.spread_open.median(),
                      "ask->bid bps/night": k.groupby("date").r_ask_bid.mean().mean() * 1e4}
    return pd.DataFrame(rows).T


def decile_table(b, sel, mask=0.5, min_ranked=300):
    """Hypothesis check (Lou, Polk & Skouras 2019): month-end deciles by trailing overnight return, held the next
    month; each decile's overnight and intraday return (gross, %/yr). b = backtest.build() output."""
    O, C, ON, S, sc, top = b["O"], b["C"], b["ON"], b["S"], b["score"], b["top"]
    ID = C / O - 1
    ID = ID.mask(ID.abs() > mask)
    me = pd.Series(S, index=S).groupby([S.year, S.month]).max().values
    dec = pd.DataFrame(np.nan, index=S, columns=ON.columns)
    for i, d in enumerate(me[:-1]):
        s = sc.loc[d].where(top.loc[d]).dropna()
        if len(s) < min_ranked:
            continue
        dec.loc[(S > d) & (S <= me[i + 1]), s.index] = (pd.qcut(s, 10, labels=False) + 1).values
    rows = {}
    for k in range(1, 11):
        o = ON.where(dec == k).mean(axis=1)[sel].mean() * 252 * 100
        i_ = ID.where(dec == k).mean(axis=1)[sel].mean() * 252 * 100
        rows[k] = {"overnight %/yr": o, "intraday %/yr": i_, "close-to-close %/yr": o + i_}
    t = pd.DataFrame(rows).T
    t.index.name = "decile"
    return t
