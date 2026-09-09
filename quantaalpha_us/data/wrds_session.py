"""A guarded, single-attempt connection; cached research needs no login."""

import os
from contextlib import suppress


def open_wrds_session(username, password):
    if os.environ.get("WRDS_DUO_READY") != "1":
        raise RuntimeError("WRDS disabled: obtain current-session Duo approval and set WRDS_DUO_READY=1.")
    if any(os.environ.get(key) for key in ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "BUILDKITE")):
        raise RuntimeError("WRDS connections are forbidden in CI, including with the approval flag.")
    import wrds
    from sqlalchemy import create_engine
    from sqlalchemy.engine import URL
    from sqlalchemy.pool import NullPool

    engine = None
    try:
        client = wrds.Connection(wrds_username=username, wrds_password=password,
                                 autoconnect=False, verbose=False)
        url = URL.create("postgresql", username=username, password=password,
                         host=wrds.sql.WRDS_POSTGRES_HOST,
                         port=wrds.sql.WRDS_POSTGRES_PORT,
                         database=wrds.sql.WRDS_POSTGRES_DB)
        engine = create_engine(url, isolation_level="AUTOCOMMIT", poolclass=NullPool,
                               connect_args=dict(wrds.sql.WRDS_CONNECT_ARGS))
        client.engine = engine
        client.connection = engine.connect()
        return client
    except Exception:
        if engine is not None:
            with suppress(Exception):
                engine.dispose()
        raise RuntimeError("WRDS connection failed after at most one attempt; no retry. Fresh approval is required.") from None
