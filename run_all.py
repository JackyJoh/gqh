"""Reproduces every number and chart in the quant note.

    python run_all.py              # in-sample backtest, then the out-of-sample test (~4 min with data cached)
    python run_all.py --backtest   # in-sample only (2010-01-04 .. 2024-10-02)
    python run_all.py --oos        # out-of-sample only (2024-10-03 .. 2026-10-02)

Missing data is downloaded first (data/download.py; needs MASSIVE_API_KEY in .env). Writes to results/:
  backtest_summary.txt   in-sample: Table 1, cutoff table, walk-forward, robustness grid, risk, hypothesis check
                         (deciles), execution check (official auction prices, spreads, break-even cost)
  backtest_daily.csv     in-sample daily returns (2 / 4 bps per side, SPY) and stocks held
  backtest_vs_spy_log.png, backtest_vs_spy_linear.png
  oos_summary.txt        out-of-sample: Table 1, monthly returns vs SPY
  oos_daily.csv          out-of-sample daily returns and stocks held
  oos_vs_spy.png
The in-sample run only loads data up to 2024-10-02. The out-of-sample period was evaluated once with the locked rules.
"""
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
from data import download as dl
from src import analysis as an, backtest as bt

RESULTS = Path(__file__).resolve().parent / "results"
T0 = time.time()
log = lambda m: print(f"[{(time.time() - T0) / 60:4.1f} min] {m}", flush=True)
pd.set_option("display.width", 220, "display.float_format", lambda v: f"{v:,.2f}")


def table1(r2, r4, spy, held):
    return {"Top 5% @2 bps/side": an.stats(r2, held), "Top 5% @4 bps/side (costs doubled)": an.stats(r4, held),
            "SPY buy & hold (no dividends)": an.stats(spy)}


def chart_lines(r2, r4, spy):
    return [("Top 5% overnight (2 bps/side)", "Top 5%", r2, "#2a78d6"),
            ("Top 5% overnight (4 bps/side, costs doubled)", "Top 5%, costs doubled", r4, "#eb6834"),
            ("SPY buy & hold", "SPY", spy, "#0b0b0b")]


def run_backtest():
    out = []
    report = lambda text="": (print(text, flush=True), out.append(text))

    log("IN-SAMPLE STEP 1/9 data, universe, signal and month-end ranks (in-sample data only)")
    b = bt.build(bt.DATA_START, bt.IS_END, log=log)
    ON, pct, S = b["ON"], b["pct"], b["S"]
    sel = S >= bt.TRADE_START
    gross, held = bt.portfolio(ON, pct)
    gross, held = gross[sel], held[sel]
    r2, r4 = bt.net(gross), bt.net(gross, 2 * bt.COST)
    spy, spy_on = b["spy"][sel].fillna(0), b["spy_on"][sel].fillna(0)

    log("IN-SAMPLE STEP 2/9 headline results (Table 1)")
    report(f"IN-SAMPLE BACKTEST  {S[sel][0].date()} .. {S[sel][-1].date()}  ({sel.sum():,} sessions)")
    report(f"data from {S[0].date()}; first ranking {S[~sel][-1].date()} uses a full 252-session window; "
           f"avg stocks held {held[held > 0].mean():.1f}")
    report("TABLE 1 (in-sample rows)\n" + an.table1_text(table1(r2, r4, spy, held)))
    yr = an.by_year(r2)
    report("\nreturn by year @2 bps: " + "  ".join(f"{y}{' (Jan-Oct 2)' if y == 2024 else ''}: {an.pct(v, 0, True)}%"
                                                   for y, v in yr.items()) + f"   -> {(yr > 0).sum()} of {len(yr)} years positive")
    report(f"turnover: the whole book is bought at every close and sold at every open = ~504x capital per year; "
           f"cost drag at 2 bps/side = {2 * bt.COST * 252 * 100:.1f}%/yr")

    log("IN-SAMPLE STEP 3/9 cutoff table (Table 2)")
    cut_gross, rows = {}, {}
    for name, lo in {"top 1%": 99, "top 2%": 98, "top 3%": 97, "top 5%": 95, "top 10%": 90}.items():
        g, n = bt.portfolio(ON, pct, lo)
        cut_gross[name] = g[sel]
        s = an.stats(bt.net(g[sel]), n[sel])
        rows[name] = {"avg stocks": s["avg stocks"], "gross overnight %/yr": g[sel].mean() * 252 * 100,
                      "CAGR %": s["CAGR %"], "Sharpe": s["Sharpe"], "max DD %": s["max DD %"], "$100k ->": s["$100k ->"],
                      "Sharpe @4bp": an.sharpe(bt.net(g[sel], 2 * bt.COST))}
    report("\nTABLE 2  cutoffs (2 bps/side)\n" + pd.DataFrame(rows).T.to_string())

    log("IN-SAMPLE STEP 4/9 walk-forward cutoff selection")
    wf, picks = bt.walk_forward(cut_gross)
    WF = {"walk-forward": an.stats(bt.net(wf))}
    WF.update({f"fixed {k}": an.stats(bt.net(v[v.index.year >= 2012])) for k, v in cut_gross.items()})
    WF["SPY buy & hold"] = an.stats(spy[spy.index.year >= 2012])
    report("\nWALK-FORWARD 2012-01 .. 2024-10 (each year: the cutoff with the best Sharpe on all prior years)")
    report("picks: " + "  ".join(f"{y}:{k}" for y, k in picks.items()))
    report(pd.DataFrame(WF).T[["CAGR %", "Sharpe", "max DD %", "$100k ->"]].to_string())

    log("IN-SAMPLE STEP 5/9 robustness grid (5 windows x 3 re-rank frequencies)")
    port, _ = bt.robustness(ON, b["top"], log=log)
    R = pd.DataFrame([{"window": w, "rerank": f, **an.stats(bt.net(r))} for (w, f), r in port.items()])
    report("\nROBUSTNESS @2 bps, 2011-07-01 .. 2024-10-02 (rows: ranking window, cols: re-ranking; locked = 12m monthly)")
    for col in ("Sharpe", "CAGR %", "max DD %"):
        grid = R.pivot(index="window", columns="rerank", values=col).reindex(["1m", "3m", "6m", "12m", "24m"])
        report(f"\n{col}\n" + grid[["daily", "weekly", "monthly"]].to_string())

    log("IN-SAMPLE STEP 6/9 risk: market exposure and decay")
    beta_on = np.polyfit(spy_on, r2, 1)[0]
    report("\nRISK (top 5% @2 bps)")
    report(f"beta to SPY daily (close to close): {np.polyfit(spy, r2, 1)[0]:.2f}")
    report(f"beta to SPY overnight (close to open): {beta_on:.2f}")
    report(f"SPY-overnight-hedged Sharpe (full-period beta, hedge costs not included): {an.sharpe(r2 - beta_on * spy_on):.2f}")
    report(f"Sharpe 2010 - 2018: {an.sharpe(r2[r2.index < '2019-01-01']):.2f}   "
           f"Sharpe 2019 - Oct 2024: {an.sharpe(r2[r2.index >= '2019-01-01']):.2f}")
    report(f"skew of daily returns: {r2.skew():.2f}")

    log("IN-SAMPLE STEP 7/9 hypothesis check: overnight vs intraday returns by decile")
    report("\nHYPOTHESIS CHECK: month-end deciles by trailing 12-month overnight return, held the next month (gross)")
    report(an.decile_table(b, sel).to_string(float_format=lambda v: f"{v:+.1f}%"))

    log("IN-SAMPLE STEP 8/9 execution check: official auction prices and bid/ask quotes on sampled nights")
    pos = bt.sample_positions(b, bt.IS_END)
    log(f"   {pos.date.nunique()} sampled nights (every {bt.SAMPLE_EVERY}th session from 2015), {len(pos):,} top-decile positions")
    AU = an.execution_table(dl.auctions_and_quotes(pos, log=log), pct)
    t5 = AU.loc["top 5%"]
    report(f"\nEXECUTION CHECK (every {bt.SAMPLE_EVERY}th session 2015-01 .. 2024-10, split nights removed; returns gross of costs)")
    report(AU.to_string(formatters={"positions": "{:,.0f}".format}))
    report(f"top 5% ({t5['positions']:,.0f} positions): {t5['auction bps/night']:.1f} bps/night at official auction prices vs "
           f"{t5['daily-bar bps/night']:.1f} with daily bars -> the backtest prices are not flattering")
    report(f"auction Sharpe @2 bps/side {t5['auction Sharpe @2bp']:.2f}; break-even cost {t5['break-even bps/side']:.1f} bps/side "
           f"= {t5['break-even bps/side'] / (bt.COST * 1e4):.1f}x the 2 bps assumed")
    report(f"median quoted spread {t5['close spread bps']:.1f} bps near the close, {t5['open spread bps']:.1f} bps at the open; "
           f"buying the ask and selling the bid instead earns {t5['ask->bid bps/night']:+.1f} bps/night -> auctions are required")

    log("IN-SAMPLE STEP 9/9 daily returns + charts")
    pd.DataFrame({"strategy_2bps": r2, "strategy_4bps": r4, "spy": spy, "stocks_held": held}).to_csv(
        RESULTS / "backtest_daily.csv", index_label="date")
    for scale in ("log", "linear"):
        an.growth_chart(chart_lines(r2, r4, spy),
                        "Growth of $100k, 2010 – Oct 2024 (in-sample): locked strategy (top 5% overnight) vs SPY",
                        "~25 stocks re-ranked monthly; buy at the closing auction, sell at the next opening auction. "
                        f"SPY excludes dividends. {scale.capitalize()} scale.",
                        RESULTS / f"backtest_vs_spy_{scale}.png", log_scale=scale == "log", year_ticks=range(2010, 2025, 2))
    (RESULTS / "backtest_summary.txt").write_text("\n".join(out) + "\n", encoding="utf-8")


def run_oos():
    out = []
    report = lambda text="": (print(text, flush=True), out.append(text))

    log(f"OUT-OF-SAMPLE STEP 1/3 data and ranks ({bt.OOS_DATA_START} .. {bt.OOS_END}; 1 year of history for the first ranking)")
    b = bt.build(bt.OOS_DATA_START, bt.OOS_END, log=log)
    S = b["S"]
    oos = (S >= bt.OOS_START) & (S <= bt.OOS_END)
    gross, held = bt.portfolio(b["ON"], b["pct"])
    r2, r4, held = bt.net(gross)[oos], bt.net(gross, 2 * bt.COST)[oos], held[oos]
    spy = b["spy"][oos].fillna(0)

    log("OUT-OF-SAMPLE STEP 2/3 results")
    mon = pd.DataFrame({"strategy_2bps": an.by_month(r2), "spy": an.by_month(spy)})
    mon.index = [f"{y}-{m:02d}" for y, m in mon.index]
    report(f"OUT-OF-SAMPLE  {S[oos][0].date()} .. {S[oos][-1].date()}  ({oos.sum()} sessions, avg stocks held {held[held > 0].mean():.1f})")
    report(f"first position: bought at the {S[S < bt.OOS_START][-1].date()} close, sold at the {S[oos][0].date()} open")
    report(f"last position:  sold at the {S[oos][-1].date()} open   (each night is booked on the day of the sell)")
    report("TABLE 1 (out-of-sample rows)\n" + an.table1_text(table1(r2, r4, spy, held)))
    w, bm = mon.strategy_2bps.idxmin(), mon.strategy_2bps.idxmax()
    report(f"\nmonths: {len(mon)}; positive {(mon.strategy_2bps > 0).sum()} ({(mon.strategy_2bps > 0).mean() * 100:.0f}%); "
           f"beat SPY {(mon.strategy_2bps > mon.spy).sum()} of {len(mon)}; "
           f"worst {w} {an.pct(mon.strategy_2bps[w], 1, True)}%; best {bm} {an.pct(mon.strategy_2bps[bm], 1, True)}%")
    report("\nMONTHLY RETURNS @2 bps vs SPY (Oct 2024 and Oct 2026 are partial months)")
    report(f"{'month':8s} {'strategy':>9s} {'SPY':>7s}  beat SPY")
    for k, a, s in zip(mon.index, mon.strategy_2bps, mon.spy):
        report(f"{k:8s} {an.pct(a, 1, True):>8s}% {an.pct(s, 1, True):>6s}%  {'yes' if a > s else 'no'}")

    log("OUT-OF-SAMPLE STEP 3/3 daily returns + chart")
    pd.DataFrame({"strategy_2bps": r2, "strategy_4bps": r4, "spy": spy, "stocks_held": held}).to_csv(
        RESULTS / "oos_daily.csv", index_label="date")
    an.growth_chart(chart_lines(r2, r4, spy), "OUT-OF-SAMPLE: growth of $100k, Oct 2024 – Oct 2026 — locked strategy vs SPY",
                    "Locked rules, evaluated once. Top 5% by trailing 12-month overnight return; buy at the closing auction, "
                    "sell at the next opening auction. SPY excludes dividends.", RESULTS / "oos_vs_spy.png")
    (RESULTS / "oos_summary.txt").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backtest", action="store_true", help="in-sample only")
    ap.add_argument("--oos", action="store_true", help="out-of-sample only")
    args = ap.parse_args()
    both = not (args.backtest or args.oos)
    RESULTS.mkdir(exist_ok=True)
    if both or args.backtest:
        run_backtest()
    if both or args.oos:
        run_oos()
    print(f"\n{'=' * 30} HEADLINE NUMBERS {'=' * 30}")
    for name in ("backtest_summary.txt", "oos_summary.txt"):
        f = RESULTS / name
        if f.exists():
            lines = f.read_text(encoding="utf-8").splitlines()
            table = lines.index(next(l for l in lines if l.startswith("TABLE 1")))
            print("\n".join(lines[:table + 5]) + "\n")
    log(f"all done -> {RESULTS}")
