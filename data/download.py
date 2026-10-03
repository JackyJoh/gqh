"""Download scripts: every piece of data the project uses, from the Massive market-data API (Polygon-style, provided by
the hackathon). Needs MASSIVE_API_KEY in .env (see .env.example). Everything is cached in data/cache/, which is never
committed (licensed data).

    python data/download.py          # download everything up front (optional: run_all.py downloads what it needs)

  1. tickers.parquet     US common stocks, active AND delisted (no survivorship bias)
  2. prices_ocv/         daily open / close / volume (split-adjusted), one file per weekday 2008-12-01 .. 2026-10-02
                         = 2009 warm-up + in-sample (2010-01-04 .. 2024-10-02) + out-of-sample (2024-10-03 .. 2026-10-02)
  3. stock_auctions/, stock_quotes/
                         official auction trades and bid/ask quotes for the execution check (in-sample sampled nights)

Resumable: every response is saved as soon as it arrives; re-run to continue or to retry failed requests.
"""
import json, hashlib, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
OCV = CACHE / "prices_ocv"
AC = CACHE / "stock_auctions"
QC = CACHE / "stock_quotes"
API = "https://api.massive.com"


def _env(name):
    if os.environ.get(name):
        return os.environ[name]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"')
    return None


def _need_key():
    if not _env("MASSIVE_API_KEY"):
        sys.exit("MASSIVE_API_KEY not set: copy .env.example to .env and add your key")


SESSION = requests.Session()
SESSION.headers["Authorization"] = f"Bearer {_env('MASSIVE_API_KEY')}"
SESSION.mount("https://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=64))   # parallel workers


def get(path, params=None, tries=5):
    url = path if path.startswith("http") else API + path
    for i in range(tries):
        try:
            r = SESSION.get(url, params=params, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** i)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


# ---------------------------------------------------------------- 1. ticker list
def tickers() -> pd.DataFrame:
    """All US common stocks (active AND delisted) that have a CIK; used to keep only common stocks (no ETFs, ADRs)."""
    f = CACHE / "tickers.parquet"
    if not f.exists():
        _need_key()
        f.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for active in ("true", "false"):
            j = get("/v3/reference/tickers", {"market": "stocks", "type": "CS", "active": active, "limit": 1000})
            while True:
                rows += j.get("results", [])
                if not j.get("next_url"):
                    break
                j = get(j["next_url"])
        df = pd.DataFrame(rows)
        df = df[df.cik.notna()]
        df = df.assign(cik=df.cik.astype(int), delisted=pd.to_datetime(df.get("delisted_utc"), utc=True).dt.tz_localize(None))
        df[["ticker", "cik", "name", "active", "delisted"]].to_parquet(f, index=False)
        print(f"   tickers: {len(df):,} common stocks with a CIK", flush=True)
    return pd.read_parquet(f)


# ---------------------------------------------------------------- 2. daily bars
def _bars_day(d):
    try:
        j = get(f"/v2/aggs/grouped/locale/us/market/stocks/{d}", {"adjusted": "true"})
    except Exception:
        j = None
    if j is None:
        return d, False
    df = pd.DataFrame(j.get("results") or [], columns=["T", "o", "c", "v"]).rename(
        columns={"T": "ticker", "o": "open", "c": "close", "v": "volume"})
    tmp = OCV / f"{d}.parquet.tmp"
    df.to_parquet(tmp, index=False)
    tmp.replace(OCV / f"{d}.parquet")            # atomic: a crash never leaves a half-written day
    return d, True


def daily_bars(start, end, workers=16, log=print):
    """Daily grouped bars for every weekday start..end. Holidays come back empty and are saved empty (correct)."""
    OCV.mkdir(parents=True, exist_ok=True)
    days = list(pd.bdate_range(start, end).strftime("%Y-%m-%d"))
    todo = [d for d in days if not (OCV / f"{d}.parquet").exists()]
    log(f"   daily bars {start} .. {end}: {len(days):,} weekdays, {len(days) - len(todo):,} cached, {len(todo):,} to download")
    if not todo:
        return
    _need_key()
    t0, failed = time.time(), []
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(_bars_day, d) for d in todo]
        for n, f in enumerate(as_completed(futs), 1):
            d, ok = f.result()
            if not ok:
                failed.append(d)
            if n % 100 == 0 or n == len(todo):
                el = time.time() - t0
                log(f"   {n:,}/{len(todo):,} ({n / len(todo):.0%})  {n / el:.1f} days/s  "
                    f"ETA {(len(todo) - n) / (n / el) / 60:.1f} min  failed {len(failed)}")
    if failed:
        sys.exit(f"{len(failed)} days failed (e.g. {sorted(failed)[:5]}); re-run the same command to retry.")


# ---------------------------------------------------------------- 3. auction trades + quotes (execution check)
_calls = {"cached": 0, "downloaded": 0}
_lock = threading.Lock()


def _count(kind):
    with _lock:
        _calls[kind] += 1


def _iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def condition_ids():
    """Trade condition codes that mark the official opening / closing auction prints."""
    f = CACHE / "auction_conditions.json"
    if not f.exists():
        _need_key()
        j = get("/v3/reference/conditions", {"asset_class": "stocks", "limit": 1000}) or {}
        trade = [c for c in j.get("results", []) if c.get("type") == "sale_condition"]   # quote codes reuse the ids
        pick = lambda names: sorted(c["id"] for c in trade if (c.get("name") or "").lower() in names)
        ids = {"open": pick({"market center official open", "market center opening trade", "opening prints"}),
               "close": pick({"market center official close", "market center closing trade", "closing prints"})}
        if not ids["open"] or not ids["close"]:
            sys.exit("could not find auction condition codes; check /v3/reference/conditions")
        f.write_text(json.dumps(ids))
    ids = json.loads(f.read_text())
    return set(ids["open"]), set(ids["close"])


def auction_price(tk, day, kind, cond_ids):
    """Closing auction: first trade at/after 16:00:00 ET with an official-close / closing-print condition.
    Opening auction: first trade at/after 09:30:00 ET with an official-open / opening-print condition.
    No flagged print in the window -> the largest trade in it (the auction cross is usually the biggest)."""
    hhmmss, span = ("16:00:00", 180) if kind == "close" else ("09:30:00", 60)
    t_et = pd.Timestamp(f"{day} {hhmmss}", tz="America/New_York")
    lo, hi = t_et.tz_convert("UTC"), (t_et + pd.Timedelta(seconds=span)).tz_convert("UTC")
    f = AC / f"{hashlib.md5(f'{tk}|{kind}|{lo}'.encode()).hexdigest()}.json"
    if f.exists():
        _count("cached")
        return json.loads(f.read_text())
    j = get(f"/v3/trades/{tk}", {"timestamp.gte": _iso(lo), "timestamp.lte": _iso(hi),
                                 "order": "asc", "sort": "timestamp", "limit": 1000})
    _count("downloaded")
    if j is None:
        return None                               # failed request: not cached, retried next run
    trades = j.get("results") or []
    out = {"price": None, "method": "none"}
    flagged = [t for t in trades if set(t.get("conditions") or []) & cond_ids]
    if flagged:
        out = {"price": flagged[0]["price"], "method": "condition"}
    elif trades:
        out = {"price": max(trades, key=lambda t: t.get("size", 0))["price"], "method": "largest"}
    f.write_text(json.dumps(out))
    return out


def quote(tk, day, hhmmss, window_s):
    """Last NBBO quote at/before hhmmss ET (looking back window_s seconds) -> (bid, ask), or None."""
    t_et = pd.Timestamp(f"{day} {hhmmss}", tz="America/New_York")
    hi, lo = t_et.tz_convert("UTC"), (t_et - pd.Timedelta(seconds=window_s)).tz_convert("UTC")
    f = QC / f"{hashlib.md5(f'{tk}|{hi}'.encode()).hexdigest()}.json"
    if f.exists():
        _count("cached")
        q = json.loads(f.read_text())
    else:
        j = get(f"/v3/quotes/{tk}", {"timestamp.lte": _iso(hi), "timestamp.gte": _iso(lo),
                                     "order": "desc", "sort": "timestamp", "limit": 1})
        _count("downloaded")
        if j is None:
            return None
        r = (j.get("results") or [None])[0]
        q = {"bid": r["bid_price"], "ask": r["ask_price"]} if r else {}
        f.write_text(json.dumps(q))
    if not q or q["bid"] <= 0 or q["ask"] <= 0 or q["ask"] < q["bid"]:
        return None
    return q["bid"], q["ask"]


def auctions_and_quotes(pos, workers=32, log=print):
    """For each position (bought at the close of `prev`, sold at the open of `date`): closing/opening auction prices
    and the quotes at 15:59:59 / 09:30:30 ET. Adds columns auc_close, auc_open, bid_c, ask_c, bid_o, ask_o."""
    AC.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)
    opens, closes = condition_ids()

    def work(r):
        p, t = r.prev.strftime("%Y-%m-%d"), r.date.strftime("%Y-%m-%d")
        return (r.Index, auction_price(r.ticker, p, "close", closes), auction_price(r.ticker, t, "open", opens),
                quote(r.ticker, p, "15:59:59", 120), quote(r.ticker, t, "09:30:30", 30))

    res, t0, last = {}, time.time(), time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(work, r) for r in pos.itertuples()]
        for n, f in enumerate(as_completed(futs), 1):
            i, *v = f.result()
            res[i] = v
            if time.time() - last > 15 or n == len(futs):
                rate = n / max(time.time() - t0, 1e-9)
                log(f"   {n:,}/{len(futs):,} positions ({n / len(futs):.0%})  ETA {(len(futs) - n) / rate / 60:.1f} min  "
                    f"[{_calls['cached']:,} cached, {_calls['downloaded']:,} downloaded]")
                last = time.time()
    g = lambda i, j, k: (res[i][j] or {}).get(k)
    q = lambda i, j, k: res[i][j][k] if res[i][j] else np.nan
    return pos.assign(auc_close=[g(i, 0, "price") for i in pos.index], auc_open=[g(i, 1, "price") for i in pos.index],
                      bid_c=[q(i, 2, 0) for i in pos.index], ask_c=[q(i, 2, 1) for i in pos.index],
                      bid_o=[q(i, 3, 0) for i in pos.index], ask_o=[q(i, 3, 1) for i in pos.index])


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    from src import backtest as bt                  # noqa: E402  (needs the ranks to know which positions to sample)
    T0 = time.time()
    log = lambda m: print(f"[{(time.time() - T0) / 60:5.1f} min] {m}", flush=True)
    log("STEP 1/3 ticker list")
    log(f"   {len(tickers()):,} common stocks -> {CACHE / 'tickers.parquet'}")
    log("STEP 2/3 daily bars")
    daily_bars(bt.DATA_START, bt.OOS_END, log=log)
    log("STEP 3/3 auction trades and quotes for the execution check (in-sample sampled nights)")
    b = bt.build(bt.DATA_START, bt.IS_END, log=log)
    auctions_and_quotes(bt.sample_positions(b, bt.IS_END), log=log)
    log(f"done -> {CACHE}")
