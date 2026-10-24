#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
US_ROOT = SCRIPT_DIR.parent
for _p in (str(US_ROOT), str(US_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quantaalpha_us.paths import resolve_from_us_root

from quantaalpha_us.trading.alpaca_rest import AlpacaRESTClient  # noqa: E402


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_client(cfg: dict) -> AlpacaRESTClient:
    broker = cfg.get("broker", {}) if isinstance(cfg.get("broker"), dict) else {}
    paper = bool(broker.get("paper", True))

    api_key = os.environ.get(str(broker.get("api_key_env", "ALPACA_PAPER_API_KEY")))
    api_secret = os.environ.get(str(broker.get("secret_key_env", "ALPACA_PAPER_API_SECRET")))
    if not api_key or not api_secret:
        raise RuntimeError("Missing Alpaca credentials in environment variables")

    base_url = broker.get("base_url")
    if not base_url:
        base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"

    timeout_seconds = float(broker.get("timeout_seconds", 30.0))
    return AlpacaRESTClient(api_key=api_key, api_secret=api_secret, base_url=base_url, timeout_seconds=timeout_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portable US kill-switch helper")
    parser.add_argument("--config", default="configs/paper_sp500.yaml")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--reason", default="manual trigger")
    parser.add_argument("--yes", action="store_true", help="Execute without interactive confirmation")
    parser.add_argument("--monitor", action="store_true", help="Run automated threshold evaluation")
    parser.add_argument("--state-file", default="data/results/trading/equity_state.json")
    return parser.parse_args()


def _apply_level(level: int, reason: str, cfg: dict, config_path: Path) -> dict[str, Any]:
    trading_dir = resolve_from_us_root("data/results/trading", US_ROOT)
    trading_dir.mkdir(parents=True, exist_ok=True)
    pause_flag = trading_dir / "PAUSE_US.flag"
    lockout_flag = trading_dir / "LOCKOUT_US.flag"

    payload: dict[str, Any] = {
        "level": level,
        "reason": reason,
        "config": str(config_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if level == 1:
        pause_flag.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["action"] = "pause_only"
        payload["pause_flag"] = str(pause_flag)
        return payload

    # Levels 2 and 3 flatten the book. Pause first, so the next trade run is
    # blocked even if the flatten call fails, and so a flattened account is not
    # immediately re-entered by the scheduler.
    pause_flag.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["pause_flag"] = str(pause_flag)

    client = _build_client(cfg)
    cancel_resp = client.cancel_all_orders()
    close_resp = client.close_all_positions()
    payload["action"] = "cancel_all_and_close_all"
    payload["cancel_response"] = cancel_resp
    payload["close_response"] = close_resp

    if level == 3:
        lockout_flag.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["lockout_flag"] = str(lockout_flag)
    return payload


def _monitor(cfg: dict, config_path: Path, state_file: Path,
             today: str | None = None) -> dict[str, Any]:
    kill_cfg = cfg.get("kill_switch", {}) if isinstance(cfg.get("kill_switch"), dict) else {}
    drawdown_cfg = kill_cfg.get("drawdown", {}) if isinstance(kill_cfg.get("drawdown"), dict) else {}
    daily_loss_cfg = kill_cfg.get("daily_loss", {}) if isinstance(kill_cfg.get("daily_loss"), dict) else {}

    drawdown_threshold = float(drawdown_cfg.get("threshold", 0.10))
    grace_days = int(drawdown_cfg.get("grace_period_trading_days", 5))
    daily_loss_threshold = float(daily_loss_cfg.get("threshold", 0.03))

    state_file.parent.mkdir(parents=True, exist_ok=True)
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    else:
        state = {}

    client = _build_client(cfg)
    account = client.get_account()
    equity = float(account.get("equity", 0.0) or 0.0)
    if equity <= 0:
        raise RuntimeError(f"Invalid account equity in monitor mode: {equity}")

    peak_equity_prev = float(state.get("peak_equity", equity) or equity)
    peak_equity = max(peak_equity_prev, equity)
    # The daily-loss reference is the last equity seen on an EARLIER date, and
    # the grace period counts dates, not runs. Rolling the reference on every
    # run measured run-to-run loss, so a loss spread over several intraday runs
    # never tripped the threshold.
    today = today or datetime.now(timezone.utc).date().isoformat()
    last_date = state.get("equity_date")
    if last_date is not None and last_date != today:
        prev_close_equity = float(state.get("equity", equity) or equity)
        days_seen = int(state.get("days_seen", 0)) + 1
    elif last_date == today:
        prev_close_equity = float(state.get("previous_close_equity", equity) or equity)
        days_seen = int(state.get("days_seen", 1))
    else:
        prev_close_equity = float(state.get("previous_close_equity", equity) or equity)
        days_seen = int(state.get("days_seen", 0)) + 1

    drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
    daily_loss = max((prev_close_equity - equity) / prev_close_equity, 0.0) if prev_close_equity > 0 else 0.0

    trigger = None
    action = None
    if drawdown > drawdown_threshold and days_seen > grace_days:
        trigger = "drawdown"
        action = _apply_level(2, f"auto trigger: drawdown={drawdown:.4f}", cfg, config_path)
    elif daily_loss > daily_loss_threshold:
        trigger = "daily_loss"
        action = _apply_level(1, f"auto trigger: daily_loss={daily_loss:.4f}", cfg, config_path)

    out_state = {
        "last_update_utc": datetime.now(timezone.utc).isoformat(),
        "equity": equity,
        "peak_equity": peak_equity,
        "equity_date": today,
        "previous_close_equity": prev_close_equity,
        "drawdown": drawdown,
        "daily_loss": daily_loss,
        "days_seen": days_seen,
    }
    state_file.write_text(json.dumps(out_state, indent=2), encoding="utf-8")

    payload = {
        "mode": "monitor",
        "config": str(config_path),
        "equity": equity,
        "peak_equity": peak_equity,
        "drawdown": drawdown,
        "drawdown_threshold": drawdown_threshold,
        "daily_loss": daily_loss,
        "daily_loss_threshold": daily_loss_threshold,
        "days_seen": days_seen,
        "triggered": trigger is not None,
        "trigger": trigger,
        "action": action,
        "state_file": str(state_file),
    }
    return payload


def main() -> None:
    args = parse_args()
    config_path = resolve_from_us_root(args.config, US_ROOT)
    cfg = _load_yaml(config_path)

    if args.monitor:
        state_file = resolve_from_us_root(args.state_file, US_ROOT)
        payload = _monitor(cfg, config_path, state_file)
        print(json.dumps(payload, indent=2))
        return

    if args.level is None:
        raise RuntimeError("Either --monitor or --level must be provided")

    if not args.yes:
        confirmation = input(f"Confirm kill-switch level {args.level} on {config_path}? [y/N]: ").strip().lower()
        if confirmation not in {"y", "yes"}:
            print(json.dumps({"status": "aborted", "reason": "not confirmed"}, indent=2))
            return

    payload = _apply_level(args.level, args.reason, cfg, config_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
