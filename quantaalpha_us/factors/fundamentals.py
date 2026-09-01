"""Point-in-time fundamental fields for the factor DSL.

Every factor the miner could previously express was a transform of price and
volume, because those were the only panels available. This adds fundamentals,
which is a much larger and less exhausted search space -- but only if the
point-in-time discipline is right, because fundamentals are where lookahead
bias is easiest to introduce and hardest to notice.

Two dates matter and conflating them is the classic error:

    datadate  the period the figures describe (e.g. quarter ending 31 March)
    rdq       the date the company actually REPORTED them

A factor may only use a figure from `rdq` onward. Aligning on `datadate` would
let the model trade on a quarter's earnings weeks before they were published,
which manufactures alpha that never existed. Everything here aligns on `rdq`,
and an extra `lag_days` buffer is applied on top so a same-day filing cannot be
traded at that day's close.

Compustat is keyed by gvkey; the daily panel is keyed by permno. The CRSP/
Compustat Merged link table maps between them, and the link itself is
time-varying, so it is applied as a date-ranged join rather than a lookup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Raw Compustat quarterly items carried through to derived fields.
_RAW_ITEMS = ("atq", "ltq", "seqq", "ceqq", "revtq", "niq", "oiadpq", "ibq",
              "cshoq", "dlttq", "dlcq")

# Trading days between a report date and the first day a factor may use it.
# One day is the minimum defensible buffer: rdq is a calendar date with no
# time of day, so a filing released after the close would otherwise be
# tradeable at that same close.
DEFAULT_LAG_DAYS = 1


def link_gvkey_to_permno(fundq: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    """Attach permno to quarterly fundamentals via the time-ranged CCM link."""
    f = fundq.copy()
    f["rdq"] = pd.to_datetime(f["rdq"], errors="coerce")
    f["datadate"] = pd.to_datetime(f["datadate"], errors="coerce")
    f = f.dropna(subset=["rdq", "gvkey"])

    l = link.copy()
    l["linkdt"] = pd.to_datetime(l["linkdt"], errors="coerce")
    # an open link has a null end date; treat it as still valid
    l["linkenddt"] = pd.to_datetime(l["linkenddt"], errors="coerce").fillna(
        pd.Timestamp.max.normalize())
    l = l.dropna(subset=["permno", "gvkey", "linkdt"])

    merged = f.merge(l[["gvkey", "permno", "linkdt", "linkenddt"]], on="gvkey", how="inner")
    # the link must be valid ON the report date, not merely for the company
    valid = (merged["rdq"] >= merged["linkdt"]) & (merged["rdq"] <= merged["linkenddt"])
    merged = merged[valid].copy()
    merged["permno"] = merged["permno"].astype(int)
    return merged


def derive_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Ratios and levels a factor expression can reference.

    Scale-free quantities only. Raw currency levels are not comparable across
    the cross-section, so a factor built on them would rank companies by size
    rather than by the property intended.
    """
    out = frame.copy()
    eps = 1e-9

    out["book_equity"] = out["ceqq"].fillna(out["seqq"])
    out["total_debt"] = out["dlttq"].fillna(0.0) + out["dlcq"].fillna(0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        out["roa"] = out["niq"] / (out["atq"] + eps)
        out["roe"] = out["niq"] / (out["book_equity"] + eps)
        out["operating_margin"] = out["oiadpq"] / (out["revtq"] + eps)
        out["leverage"] = out["total_debt"] / (out["atq"] + eps)
        out["asset_turnover"] = out["revtq"] / (out["atq"] + eps)
        out["book_per_share"] = out["book_equity"] / (out["cshoq"] + eps)
        out["earnings_per_share"] = out["ibq"] / (out["cshoq"] + eps)

    # accruals proxy: earnings not backed by operating income
    with np.errstate(divide="ignore", invalid="ignore"):
        out["accrual_gap"] = (out["niq"] - out["oiadpq"]) / (out["atq"] + eps)

    numeric = [c for c in out.columns if c not in ("gvkey", "permno", "rdq", "datadate")]
    out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)
    return out


def build_pit_panel(
    fundq: pd.DataFrame,
    link: pd.DataFrame,
    dates: pd.DatetimeIndex,
    permnos: pd.Index | None = None,
    lag_days: int = DEFAULT_LAG_DAYS,
    fields: tuple[str, ...] = ("roa", "roe", "operating_margin", "leverage",
                               "asset_turnover", "book_per_share",
                               "earnings_per_share", "accrual_gap"),
) -> dict[str, pd.DataFrame]:
    """Forward-filled date x permno panels, each usable only after its report date.

    Construction is deliberately conservative at every step: rows are stamped
    with `available_from = rdq + lag_days`, reindexed onto the trading calendar,
    and forward-filled. A value therefore first appears the day AFTER it was
    reported and persists until the next report supersedes it, which is exactly
    what an investor could have known.
    """
    linked = link_gvkey_to_permno(fundq, link)
    derived = derive_fields(linked)

    derived["available_from"] = derived["rdq"] + pd.Timedelta(days=lag_days)
    derived = derived.sort_values(["permno", "available_from", "datadate"])
    # a restatement can republish an earlier quarter; keep the latest report
    derived = derived.drop_duplicates(subset=["permno", "available_from"], keep="last")

    if permnos is not None:
        derived = derived[derived["permno"].isin(set(int(p) for p in permnos))]

    panels: dict[str, pd.DataFrame] = {}
    for field in fields:
        if field not in derived.columns:
            continue
        wide = derived.pivot_table(index="available_from", columns="permno",
                                   values=field, aggfunc="last")
        wide = wide.reindex(wide.index.union(dates)).sort_index().ffill()
        # pin to float64: Compustat columns arrive as pandas nullable dtypes,
        # on which numpy ufuncs raise "boolean value of NA is ambiguous"
        panels[field] = wide.reindex(dates).astype("float64")
    return panels


def attach_to_bars(bars: pd.DataFrame, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Melt permno-keyed panels back onto a long (date, symbol, permno) bar frame."""
    out = bars.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "permno" not in out.columns:
        raise ValueError("bars must carry permno to attach fundamentals")
    out["permno"] = pd.to_numeric(out["permno"], errors="coerce").astype("Int64")

    index = pd.MultiIndex.from_arrays([out["date"], out["permno"].astype("float")])
    for field, panel in panels.items():
        stacked = panel.stack(future_stack=True)
        stacked.index = stacked.index.set_levels(
            stacked.index.levels[1].astype(float), level=1)
        out[field] = stacked.reindex(index).to_numpy()
    return out
