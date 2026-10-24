from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class ValidationReport:
    passed: bool
    min_count: int
    max_count: int
    violations: int
    details: str


class SP500Universe:
    """Point-in-time S&P 500 universe provider."""

    def __init__(self, membership_path: str, ticker_mapping_file: Optional[str] = None) -> None:
        self.membership_path = Path(membership_path)
        self.membership = self._load_membership(self.membership_path)
        self.membership = self.membership.copy().assign(
            date=pd.to_datetime(self.membership["date"], errors="coerce").dt.normalize(),
            symbol=self.membership["symbol"].astype(str).str.upper(),
        )
        if "active" not in self.membership.columns:
            self.membership = self.membership.assign(active=True)
        self.membership = self.membership.assign(active=self.membership["active"].astype(bool))

        self.membership = self.membership.dropna(subset=["date", "symbol"])
        self.membership = self.membership.sort_values(["date", "symbol"]).reset_index(drop=True)

        self._by_date = {
            date: group[group["active"]]["symbol"].tolist()
            for date, group in self.membership.groupby("date", sort=True)
        }
        # A CRSP-built membership file (it carries permno) already holds the
        # ticker that was valid on each date, which is what the bars carry.
        # The ticker map is undated and keyed by old symbol, so applying it
        # would rename a member to a ticker it only adopted later (dropping it
        # from the universe before the rename, which is look-ahead) and would
        # merge companies that reused a ticker. It is applied only to
        # membership files without permno.
        self.dated_symbols = "permno" in self.membership.columns
        self.ticker_map = {} if self.dated_symbols else self._load_ticker_mapping(ticker_mapping_file)

    @staticmethod
    def _load_membership(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Membership file not found: {path}")
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    @staticmethod
    def _load_ticker_mapping(path: Optional[str]) -> dict[str, str]:
        if not path:
            return {}
        p = Path(path)
        if not p.exists():
            return {}

        df = pd.read_csv(p)
        old_col = next((c for c in ["old_symbol", "from_symbol", "old", "symbol_old"] if c in df.columns), None)
        new_col = next((c for c in ["new_symbol", "to_symbol", "new", "symbol_new"] if c in df.columns), None)
        if old_col is None or new_col is None:
            return {}

        mapping: dict[str, str] = {}
        for row in df.itertuples(index=False):
            old_symbol = getattr(row, old_col)
            new_symbol = getattr(row, new_col)
            if pd.notna(old_symbol) and pd.notna(new_symbol):
                mapping[str(old_symbol).upper()] = str(new_symbol).upper()
        return mapping

    def get_members(self, date: str | pd.Timestamp) -> list[str]:
        target = pd.Timestamp(date).normalize()
        members = self._by_date.get(target, [])
        if not self.ticker_map:
            return sorted(set(members))
        mapped = [self.ticker_map.get(symbol, symbol) for symbol in members]
        return sorted(set(mapped))

    def get_members_range(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> dict[pd.Timestamp, list[str]]:
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        days = sorted(d for d in self._by_date.keys() if start_ts <= d <= end_ts)
        return {d: self.get_members(d) for d in days}

    def validate(self, min_members: int = 480, max_members: int = 510) -> ValidationReport:
        counts = self.membership[self.membership["active"]].groupby("date")["symbol"].nunique()
        if counts.empty:
            return ValidationReport(
                passed=False,
                min_count=0,
                max_count=0,
                violations=1,
                details="No active membership rows available",
            )

        violations = counts[(counts < min_members) | (counts > max_members)]
        details = (
            "Membership count bounds satisfied"
            if violations.empty
            else f"Out-of-range dates={len(violations)} sample={violations.head(5).to_dict()}"
        )
        return ValidationReport(
            passed=violations.empty,
            min_count=int(counts.min()),
            max_count=int(counts.max()),
            violations=int(len(violations)),
            details=details,
        )

    def to_membership_mask(self, dates: pd.Index, symbols: list[str]) -> pd.DataFrame:
        """Build date x symbol boolean mask for backtest filtering."""
        idx = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
        cols = [str(s).upper() for s in symbols]

        rows: list[dict[str, bool]] = []
        for d in idx:
            active = set(self.get_members(d))
            rows.append({s: s in active for s in cols})

        mask = pd.DataFrame(rows, index=idx, columns=cols)
        mask.index = pd.DatetimeIndex(dates)
        return mask.astype(bool)
