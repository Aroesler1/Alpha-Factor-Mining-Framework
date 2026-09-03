from __future__ import annotations

from .crsp_client import CRSPClient


def build_market_data_client(*, source: str = "auto") -> CRSPClient:
    """Return the market-data client. CRSP via WRDS is the only source.

    This used to multiplex CRSP against an EODHD fallback. EODHD is gone: every
    call site the pipeline uses (daily bars, bulk daily bars, ticker mapping) is
    served by CRSPClient, which additionally carries PERMNO identity, delisting
    returns and point-in-time membership that a plain vendor EOD feed does not.

    `source` is retained so existing callers and scripts keep working, but the
    only accepted values are "auto" and "crsp".
    """
    source = str(source).strip().lower()
    if source not in {"auto", "crsp"}:
        raise ValueError(
            f"Unknown market-data source {source!r}. CRSP is the only source; "
            'use "crsp" or "auto".'
        )

    client = CRSPClient()
    if not client.is_configured():
        raise RuntimeError(
            "CRSP credentials are not configured. Set CRSP_USERNAME and "
            "CRSP_API_KEY (or WRDS_USERNAME / WRDS_PASSWORD)."
        )
    return client
