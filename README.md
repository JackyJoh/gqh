# Overnight Drift — Gator Quant Hacks 2026 (Systematic Trading)

Stocks that have earned a lot **overnight** (close → next open) over the past year keep doing it. Every trading day we
buy the top 5% of the 500 most-traded US stocks by that measure in the **closing auction** and sell them in the next
**opening auction**. We hold nothing during the day.

Based on Lou, Polk & Skouras (2019), *A tug of war: overnight versus intraday expected returns*, JFE 134(1).

## Reproduce the results

> **Data download required:** Market data is not included in this repo. Download it directly from Massive using your own API key: set `MASSIVE_API_KEY` in `.env`, then run `python data/download.py`. Your key must have access to the required endpoints. Downloads stay local in `data/cache/`.

**1. Install** (Python 3.11+; tested on 3.13.7)
```bash
pip install -r requirements.txt
```

**2. Add your API key.** Data comes from the Massive market-data API (Polygon-style, provided by the hackathon).
```bash
cp .env.example .env        # Windows: copy .env.example .env
                            # then put your key in .env:  MASSIVE_API_KEY=...
```

**3. Run**
```bash
python run_all.py
```
Downloads any missing data into `data/cache/` (resumable: re-run if interrupted), runs the in-sample backtest and the
out-of-sample test, writes every number and chart in the quant note to `results/`, and prints the headline numbers.

| Command | What it does | Time with data cached |
|---|---|---|
| `python run_all.py` | in-sample, then out-of-sample | ~4 min |
| `python run_all.py --backtest` | in-sample only (2010-01-04 → 2024-10-02) | ~3.5 min |
| `python run_all.py --oos` | out-of-sample only (2024-10-03 → 2026-10-02) | ~15 s |
| `python data/download.py` | optional: download all data up front | — |

**First run from an empty cache:** downloads ~1.1 GB: daily bars for ~4,650 weekdays plus
~97k auction-trade and quote lookups for the execution check. Download time depends on API access and rate limits.
Everything is cached locally, so later runs take the times above. Licensed market data is not included in this
repository. Use your own Massive API key with access to the required endpoints; raw data, caches, and data archives
must remain local and must not be committed. The optional `python data/download.py` command downloads the data
ahead of running the backtest.

## Expected output

Net of 2 bps per side; SPY excludes dividends.

| Period | $100k → | CAGR | Sharpe | Max DD |
|---|---|---|---|---|
| In-sample 2010-01-04 → 2024-10-02 | $1,746,184 | 21.4% | 1.10 | −36.5% |
| In-sample, costs doubled (4 bps/side) | $395,885 | 9.8% | 0.58 | −51.3% |
| SPY, in-sample | $510,463 | 11.7% | 0.73 | −34.1% |
| **Out-of-sample 2024-10-03 → 2026-10-02** | **$252,942** | **59.5%** | **1.40** | **−34.3%** |
| Out-of-sample, costs doubled | $207,065 | 44.2% | 1.14 | −35.6% |
| SPY, out-of-sample | $135,295 | 16.4% | 1.00 | −19.0% |

| File in `results/` | Contents |
|---|---|
| `backtest_summary.txt` | Table 1 (in-sample), cutoff table (top 1/2/3/5/10%), walk-forward cutoff choice, robustness grid (ranking window × re-rank frequency), risk (betas, hedged Sharpe, pre/post-2019), hypothesis check (overnight vs intraday by decile), execution check (official auction prices, spreads, break-even cost) |
| `oos_summary.txt` | Table 1 (out-of-sample), monthly returns vs SPY |
| `backtest_daily.csv`, `oos_daily.csv` | daily returns (2 and 4 bps/side, SPY) and stocks held, to recompute any statistic |
| `backtest_vs_spy_log.png`, `backtest_vs_spy_linear.png`, `oos_vs_spy.png` | growth of $100k vs SPY |

## The strategy

| | Rule |
|---|---|
| **Universe** | each day, the 500 US common stocks with the highest 20-day median dollar volume, price ≥ $5; point-in-time, including later-delisted stocks (no survivorship bias) |
| **Signal** | mean overnight return (open ÷ previous close − 1) over the last 252 trading days, ≥ 200 valid days |
| **Ranking** | at each month-end close, percentile-rank the universe; skip the month if fewer than 300 stocks have a signal |
| **Trade** | every trading day of the next month, buy the stocks above the 95th percentile (~25) with market-on-close orders and sell them with market-on-open orders; equal weight; flat during the day |
| **Costs** | 2 bps per side (each buy and each sell), all-in; 4 bps per side as the costs-doubled stress test |

All rule constants are at the top of `src/signals.py` and `src/backtest.py`.

**Data and dates.** Daily bars start 2008-12-01, so the first ranking (2009-12-31) uses a full year of history and
trading starts 2010-01-04. The out-of-sample period is the last 2 years (track rule: last 20% or 2 years, whichever is
shorter). All rules were chosen on in-sample data and locked; the out-of-sample test was run once and nothing was
changed afterwards. The in-sample run only loads data up to 2024-10-02.

## Repo layout

```
README.md          setup + one command to run
requirements.txt   pinned dependencies
.env.example       API key template (.env is never committed)
data/
  download.py      all data download code
  cache/           downloaded market data (local only, ignored by Git)
src/
  signals.py       overnight returns, universe, score, month-end percentile ranks
  backtest.py      rules and dates, price matrices, portfolio returns, walk-forward, robustness
  analysis.py      statistics, tables, charts
run_all.py         reproduces the note
results/           outputs, committed so you can compare with your run
```

No API keys are in this repo.
