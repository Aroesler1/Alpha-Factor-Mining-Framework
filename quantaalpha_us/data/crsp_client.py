from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import pandas as pd


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ResolvedTable:
    library: str
    table: str
    columns: set[str]


def _clean_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _sic_to_sector(sic: Any) -> str:
    try:
        code = int(float(sic))
    except Exception:
        return "Unknown"

    if 100 <= code <= 999:
        return "Energy"
    if 1000 <= code <= 1499:
        return "Materials"
    if 1500 <= code <= 1799:
        return "Industrials"
    if 2000 <= code <= 2399:
        return "Consumer Staples"
    if 2400 <= code <= 3999:
        if 3570 <= code <= 3699 or 3810 <= code <= 3839 or 7370 <= code <= 7379:
            return "Information Technology"
        return "Industrials"
    if 4000 <= code <= 4799:
        return "Industrials"
    if 4800 <= code <= 4899:
        return "Communication Services"
    if 4900 <= code <= 4949:
        return "Utilities"
    if 5000 <= code <= 5199:
        return "Industrials"
    if 5200 <= code <= 5999:
        return "Consumer Discretionary"
    if 6000 <= code <= 6499:
        return "Financials"
    if 6500 <= code <= 6999:
        return "Real Estate"
    if 7000 <= code <= 7299:
        return "Consumer Discretionary"
    if 7300 <= code <= 7399:
        return "Industrials"
    if 7500 <= code <= 7999:
        return "Consumer Discretionary"
    if 8000 <= code <= 8099:
        return "Health Care"
    if 8100 <= code <= 8999:
        return "Industrials"
    if 9100 <= code <= 9729:
        return "Utilities"
    return "Unknown"


class CRSPClient:
    """WRDS/CRSP client for historical S&P 500 membership and daily stock bars."""

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        username_env: str = "CRSP_USERNAME",
        password_envs: Optional[Iterable[str]] = None,
    ) -> None:
        password_keys = tuple(password_envs or ("CRSP_API_KEY", "CRSP_PASSWORD", "WRDS_PASSWORD"))
        self.username = username or os.environ.get(username_env) or os.environ.get("WRDS_USERNAME")
        self.password = password or next((os.environ.get(k) for k in password_keys if os.environ.get(k)), None)
        self._db: Any = None
        self._connection_attempted = False
        self._libraries: Optional[list[str]] = None
        self._table_cache: dict[str, list[str]] = {}
        self._column_cache: dict[tuple[str, str], set[str]] = {}

    @property
    def source_name(self) -> str:
        return "crsp"

    def is_configured(self) -> bool:
        return bool(self.username and self.password)

    @contextmanager
    def _password_env(self):
        old_pg = os.environ.get("PGPASSWORD")
        if self.password:
            os.environ["PGPASSWORD"] = self.password
        try:
            yield
        finally:
            if old_pg is None:
                os.environ.pop("PGPASSWORD", None)
            else:
                os.environ["PGPASSWORD"] = old_pg

    def close(self) -> None:
        if self._db is not None:
            try:
                self._db.close()
            except Exception:
                logger.debug("Failed closing WRDS connection", exc_info=True)
            self._db = None

    def _get_db(self):
        if self._db is not None:
            return self._db
        if os.environ.get("WRDS_DUO_READY") != "1":
            raise RuntimeError("WRDS disabled: obtain current-session Duo approval and set WRDS_DUO_READY=1.")
        if self._connection_attempted:
            raise RuntimeError("This client already attempted WRDS; no retry. Fresh approval and a new client are required.")
        if not self.is_configured():
            raise RuntimeError("CRSP is not configured. Set CRSP_USERNAME and CRSP_API_KEY (or CRSP_PASSWORD).")
        from quantaalpha_us.data.wrds_session import open_wrds_session

        self._connection_attempted = True
        self._db = open_wrds_session(self.username, self.password)
        return self._db

    def _list_libraries(self) -> list[str]:
        if self._libraries is None:
            libs = self._get_db().list_libraries()
            self._libraries = [str(x) for x in libs]
        return self._libraries

    def _list_tables(self, library: str) -> list[str]:
        if library not in self._table_cache:
            self._table_cache[library] = [str(x) for x in self._get_db().list_tables(library=library)]
        return self._table_cache[library]

    def _describe_columns(self, library: str, table: str) -> set[str]:
        key = (library, table)
        if key not in self._column_cache:
            desc = self._get_db().describe_table(library=library, table=table)
            self._column_cache[key] = {str(x).lower() for x in desc["name"].tolist()}
        return self._column_cache[key]

    def _resolve_table(self, candidates: list[tuple[str, str]]) -> Optional[_ResolvedTable]:
        for library, table in candidates:
            if library not in self._list_libraries():
                continue
            tables = self._list_tables(library)
            if table not in tables:
                continue
            return _ResolvedTable(library=library, table=table, columns=self._describe_columns(library, table))
        return None

    def _stocknames_table(self) -> _ResolvedTable:
        resolved = self._resolve_table(
            [
                ("crsp", "stocknames_v2"),
                ("crsp", "stocknames"),
                ("crsp", "msenames"),
            ]
        )
        if resolved is None:
            raise RuntimeError("Could not find a CRSP stocknames table in WRDS.")
        return resolved

    def _daily_table(self) -> _ResolvedTable:
        resolved = self._resolve_table(
            [
                ("crsp", "dsf_v2"),
                ("crsp", "dsf"),
            ]
        )
        if resolved is None:
            raise RuntimeError("Could not find a CRSP daily stock file table in WRDS.")
        return resolved

    def _sp500_membership_table(self) -> Optional[_ResolvedTable]:
        exact = self._resolve_table(
            [
                ("crsp_a_indexes", "dsp500list_v2"),
                ("crsp_a_indexes", "dsp500list"),
                ("crsp_q_indexes", "dsp500list_v2"),
                ("crsp_q_indexes", "dsp500list"),
                ("crsp", "dsp500list_v2"),
                ("crsp", "dsp500list"),
            ]
        )
        if exact is not None:
            return exact

        candidate_libraries = [lib for lib in self._list_libraries() if lib.startswith("crsp")]
        for library in candidate_libraries:
            for table in self._list_tables(library):
                lower = table.lower()
                if "sp500" not in lower and "s&p" not in lower:
                    continue
                if "list" not in lower and "const" not in lower and "member" not in lower:
                    continue
                return _ResolvedTable(library=library, table=table, columns=self._describe_columns(library, table))
        return None

    @staticmethod
    def _pick_first(columns: set[str], names: Iterable[str]) -> Optional[str]:
        for name in names:
            if name in columns:
                return name
        return None

    @staticmethod
    def _date_sql(value: Optional[str]) -> str:
        return str(pd.Timestamp(value).date()) if value else str(pd.Timestamp("1900-01-01").date())

    @staticmethod
    def _first_available(columns: set[str], names: Iterable[str]) -> Optional[str]:
        for name in names:
            if name in columns:
                return name
        return None

    def _daily_sql_parts(self, columns: set[str]) -> dict[str, str]:
        date_col = self._first_available(columns, ["date", "dlycaldt"])
        if date_col is None:
            raise RuntimeError("CRSP daily table is missing a supported date column.")

        close_col = self._first_available(columns, ["prc", "dlyprc", "close", "dlyclose"])
        open_col = self._first_available(columns, ["openprc", "dlyopen", "open"])
        high_col = self._first_available(columns, ["askhi", "dlyhigh", "high"])
        low_col = self._first_available(columns, ["bidlo", "dlylow", "low"])
        vol_col = self._first_available(columns, ["vol", "dlyvol", "volume"])
        adj_factor_col = self._first_available(columns, ["cfacpr", "dlycumfacpr", "dlyfacprc"])

        if close_col is None:
            raise RuntimeError("CRSP daily table is missing a supported close-price column.")

        close_expr = f"ABS(d.{close_col})"
        open_expr = f"ABS(d.{open_col})" if open_col else close_expr
        high_expr = f"ABS(d.{high_col})" if high_col else close_expr
        low_expr = f"ABS(d.{low_col})" if low_col else close_expr
        volume_expr = f"COALESCE(d.{vol_col}, 0)" if vol_col else "0"
        if adj_factor_col:
            adj_expr = (
                f"CASE WHEN d.{adj_factor_col} IS NULL OR d.{adj_factor_col} = 0 THEN {close_expr} "
                f"ELSE {close_expr} / ABS(d.{adj_factor_col}) END"
            )
        else:
            adj_expr = close_expr
        return {
            "date_col": date_col,
            "close_expr": close_expr,
            "open_expr": open_expr,
            "high_expr": high_expr,
            "low_expr": low_expr,
            "volume_expr": volume_expr,
            "adj_expr": adj_expr,
        }

    def _history_query(
        self,
        *,
        symbol: str,
        permno: Optional[int],
        from_date: Optional[str],
        to_date: Optional[str],
    ) -> str:
        daily = self._daily_table()
        names = self._stocknames_table()
        parts = self._daily_sql_parts(daily.columns)

        join = (
            f"LEFT JOIN {names.library}.{names.table} n "
            "ON d.permno = n.permno "
            f"AND d.{parts['date_col']} >= n.namedt "
            f"AND d.{parts['date_col']} <= COALESCE(n.nameenddt, DATE '9999-12-31')"
        )
        symbol_sql = _clean_symbol(symbol).replace("'", "''")
        filters = [f"d.{parts['date_col']} >= DATE '{self._date_sql(from_date)}'"]
        if to_date:
            filters.append(f"d.{parts['date_col']} <= DATE '{self._date_sql(to_date)}'")
        if permno is not None:
            filters.append(f"CAST(d.permno AS BIGINT) = {int(permno)}")
        else:
            filters.append(f"UPPER(COALESCE(n.ticker, '')) = '{symbol_sql}'")

        return f"""
            SELECT
                d.{parts['date_col']} AS date,
                UPPER(COALESCE(NULLIF(TRIM(n.ticker), ''), '{symbol_sql}')) AS symbol,
                {parts['open_expr']} AS open,
                {parts['high_expr']} AS high,
                {parts['low_expr']} AS low,
                {parts['close_expr']} AS close,
                {parts['adj_expr']} AS adj_close,
                {parts['volume_expr']} AS volume,
                CAST(d.permno AS BIGINT) AS permno
            FROM {daily.library}.{daily.table} d
            {join}
            WHERE {" AND ".join(filters)}
            ORDER BY d.{parts['date_col']}
        """

    def get_eod_history(
        self,
        symbol: str,
        exchange: str = "US",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        *,
        use_cache: bool = False,
        permno: Optional[int] = None,
    ) -> pd.DataFrame:
        _ = exchange, use_cache
        sql = self._history_query(symbol=symbol, permno=permno, from_date=from_date, to_date=to_date)
        df = self._get_db().raw_sql(sql)
        if df.empty:
            return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "adj_close", "volume", "permno"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df["symbol"] = df["symbol"].astype(str).str.upper()
        for col in ["open", "high", "low", "close", "adj_close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if "permno" in df.columns:
            df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
        return df[["date", "symbol", "open", "high", "low", "close", "adj_close", "volume", "permno"]].dropna(
            subset=["date", "symbol"]
        )

    def get_eod_history_batch(
        self,
        requests: list[dict[str, Any]],
        *,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> pd.DataFrame:
        if not requests:
            return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "adj_close", "volume", "permno"])

        daily = self._daily_table()
        names = self._stocknames_table()
        parts = self._daily_sql_parts(daily.columns)

        permnos = sorted({int(item["permno"]) for item in requests if item.get("permno") is not None and pd.notna(item.get("permno"))})
        if not permnos:
            frames = [
                self.get_eod_history(
                    symbol=str(item["symbol"]),
                    from_date=from_date,
                    to_date=to_date,
                    permno=None,
                )
                for item in requests
            ]
            frames = [df for df in frames if not df.empty]
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
                columns=["date", "symbol", "open", "high", "low", "close", "adj_close", "volume", "permno"]
            )

        permno_list = ", ".join(str(p) for p in permnos)
        filters = [
            f"CAST(d.permno AS BIGINT) IN ({permno_list})",
            f"d.{parts['date_col']} >= DATE '{self._date_sql(from_date)}'",
        ]
        if to_date:
            filters.append(f"d.{parts['date_col']} <= DATE '{self._date_sql(to_date)}'")
        sql = f"""
            SELECT
                d.{parts['date_col']} AS date,
                UPPER(COALESCE(NULLIF(TRIM(n.ticker), ''), CAST(CAST(d.permno AS BIGINT) AS TEXT))) AS symbol,
                {parts['open_expr']} AS open,
                {parts['high_expr']} AS high,
                {parts['low_expr']} AS low,
                {parts['close_expr']} AS close,
                {parts['adj_expr']} AS adj_close,
                {parts['volume_expr']} AS volume,
                CAST(d.permno AS BIGINT) AS permno
            FROM {daily.library}.{daily.table} d
            LEFT JOIN {names.library}.{names.table} n
              ON d.permno = n.permno
             AND d.{parts['date_col']} >= n.namedt
             AND d.{parts['date_col']} <= COALESCE(n.nameenddt, DATE '9999-12-31')
            WHERE {" AND ".join(filters)}
            ORDER BY d.permno, d.{parts['date_col']}
        """
        df = self._get_db().raw_sql(sql)
        if df.empty:
            return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "adj_close", "volume", "permno"])

        df = df.assign(
            date=pd.to_datetime(df["date"], errors="coerce").dt.normalize(),
            permno=pd.to_numeric(df["permno"], errors="coerce").astype("Int64"),
        )
        df = df.assign(
            symbol=df["symbol"].astype(str).str.upper()
        )
        df.loc[df["symbol"].isin(["", "NAN", "NONE"]), "symbol"] = df.loc[
            df["symbol"].isin(["", "NAN", "NONE"]), "permno"
        ].astype(str)
        for col in ["open", "high", "low", "close", "adj_close", "volume"]:
            df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")
        return df[["date", "symbol", "open", "high", "low", "close", "adj_close", "volume", "permno"]].dropna(
            subset=["date", "symbol"]
        )

    def get_bulk_eod(
        self,
        exchange: str = "US",
        date: Optional[str] = None,
        *,
        use_cache: bool = False,
    ) -> pd.DataFrame:
        _ = exchange, use_cache
        daily = self._daily_table()
        names = self._stocknames_table()
        parts = self._daily_sql_parts(daily.columns)
        target_date = self._date_sql(date or str(pd.Timestamp.today().date()))

        sql = f"""
            SELECT
                d.{parts['date_col']} AS date,
                UPPER(COALESCE(NULLIF(TRIM(n.ticker), ''), CAST(CAST(d.permno AS BIGINT) AS TEXT))) AS symbol,
                {parts['open_expr']} AS open,
                {parts['high_expr']} AS high,
                {parts['low_expr']} AS low,
                {parts['close_expr']} AS close,
                {parts['adj_expr']} AS adj_close,
                {parts['volume_expr']} AS volume,
                CAST(d.permno AS BIGINT) AS permno
            FROM {daily.library}.{daily.table} d
            LEFT JOIN {names.library}.{names.table} n
              ON d.permno = n.permno
             AND d.{parts['date_col']} >= n.namedt
             AND d.{parts['date_col']} <= COALESCE(n.nameenddt, DATE '9999-12-31')
            WHERE d.{parts['date_col']} = DATE '{target_date}'
        """
        df = self._get_db().raw_sql(sql)
        if df.empty:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df["symbol"] = df["symbol"].astype(str).str.upper()
        for col in ["open", "high", "low", "close", "adj_close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if "permno" in df.columns:
            df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
        return df

    def get_sp500_constituents_historical(self, *, use_cache: bool = True) -> pd.DataFrame:
        _ = use_cache
        membership = self._sp500_membership_table()
        if membership is None:
            return pd.DataFrame(
                columns=[
                    "Code",
                    "Exchange",
                    "Name",
                    "Sector",
                    "Industry",
                    "StartDate",
                    "EndDate",
                    "IsActiveNow",
                    "IsDelisted",
                    "permno",
                ]
            )

        names = self._stocknames_table()
        permno_col = self._pick_first(membership.columns, ["permno", "lpermno"])
        start_col = self._pick_first(membership.columns, ["mbrstartdt", "start", "startdt", "from", "fromdt", "begdt"])
        end_col = self._pick_first(membership.columns, ["mbrenddt", "ending", "end", "enddt", "thru", "thrudt"])
        if permno_col is None or start_col is None:
            logger.warning("CRSP S&P 500 membership table %s.%s has unsupported columns", membership.library, membership.table)
            return pd.DataFrame()

        end_expr = f"COALESCE(m.{end_col}, DATE '9999-12-31')" if end_col else "DATE '9999-12-31'"
        sic_expr = "n.siccd" if "siccd" in names.columns else "NULL"
        name_expr = "n.comnam" if "comnam" in names.columns else "n.ticker"
        sql = f"""
            SELECT
                CAST(m.{permno_col} AS BIGINT) AS permno,
                UPPER(NULLIF(TRIM(n.ticker), '')) AS code,
                {name_expr} AS name,
                {sic_expr} AS siccd,
                GREATEST(m.{start_col}, n.namedt) AS startdate,
                LEAST({end_expr}, COALESCE(n.nameenddt, DATE '9999-12-31')) AS enddate
            FROM {membership.library}.{membership.table} m
            JOIN {names.library}.{names.table} n
              ON m.{permno_col} = n.permno
             AND n.namedt <= {end_expr}
             AND COALESCE(n.nameenddt, DATE '9999-12-31') >= m.{start_col}
            WHERE NULLIF(TRIM(n.ticker), '') IS NOT NULL
        """
        df = self._get_db().raw_sql(sql).copy()
        if df.empty:
            return pd.DataFrame()

        code = df["code"].astype(str).str.upper()
        end_dates = pd.to_datetime(df["enddate"], errors="coerce").dt.normalize()
        df = df.assign(
            Code=code,
            Exchange="US",
            Name=df["name"].fillna(code).astype(str),
            Sector=df["siccd"].map(_sic_to_sector),
            Industry=df["siccd"].apply(lambda x: f"SIC {int(float(x))}" if pd.notna(x) else "Unknown"),
            StartDate=pd.to_datetime(df["startdate"], errors="coerce").dt.normalize(),
            EndDate=end_dates,
            IsActiveNow=end_dates.isna() | (end_dates >= pd.Timestamp.today().normalize()),
            IsDelisted=False,
            permno=pd.to_numeric(df["permno"], errors="coerce").astype("Int64"),
        )
        df = df[df["Code"].str.len() > 0].copy()
        df = df[df["EndDate"].isna() | df["StartDate"].isna() | (df["EndDate"] >= df["StartDate"])]
        return df[
            [
                "Code",
                "Exchange",
                "Name",
                "Sector",
                "Industry",
                "StartDate",
                "EndDate",
                "IsActiveNow",
                "IsDelisted",
                "permno",
            ]
        ].drop_duplicates()

    def get_ticker_mapping(self, *, use_cache: bool = True) -> pd.DataFrame:
        _ = use_cache
        names = self._stocknames_table()
        name_expr = "namedt" if "namedt" in names.columns else "date"
        sql = f"""
            SELECT
                CAST(permno AS BIGINT) AS permno,
                UPPER(NULLIF(TRIM(ticker), '')) AS ticker,
                {name_expr} AS namedt
            FROM {names.library}.{names.table}
            WHERE NULLIF(TRIM(ticker), '') IS NOT NULL
            ORDER BY permno, {name_expr}
        """
        df = self._get_db().raw_sql(sql).copy()
        if df.empty:
            return pd.DataFrame(columns=["old_symbol", "new_symbol", "effective_date", "reason"])

        df = df.assign(
            ticker=df["ticker"].astype(str).str.upper(),
            namedt=pd.to_datetime(df["namedt"], errors="coerce").dt.normalize(),
        )
        rows: list[dict[str, Any]] = []
        for _, group in df.groupby("permno", sort=True):
            prev_ticker: Optional[str] = None
            for row in group.itertuples(index=False):
                ticker = _clean_symbol(row.ticker)
                if not ticker:
                    continue
                if prev_ticker and ticker != prev_ticker:
                    rows.append(
                        {
                            "old_symbol": prev_ticker,
                            "new_symbol": ticker,
                            "effective_date": str(pd.Timestamp(row.namedt).date()) if pd.notna(row.namedt) else None,
                            "reason": "crsp_stocknames_permno",
                        }
                    )
                prev_ticker = ticker

        if not rows:
            return pd.DataFrame(columns=["old_symbol", "new_symbol", "effective_date", "reason"])
        out = pd.DataFrame(rows).drop_duplicates(subset=["old_symbol", "new_symbol", "effective_date"])
        return out.reset_index(drop=True)
