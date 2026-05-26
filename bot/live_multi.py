"""Multi-pair launcher for Axiom Omega.

Spawns one ``python -m bot.live --config <per-pair config>`` subprocess per
pair, supervises them, and restarts any child that exits unexpectedly. Each
child is a fully isolated LiveBot:

  * Its own MT5 connection (the MetaTrader5 Python package supports multiple
    concurrent terminal calls from the same process group; running each
    LiveBot as a separate process side-steps any GIL / re-entry concerns)
  * Its own paper-positions file (filename embeds the symbol)
  * Its own ML model file (``ml.model_path`` differs per config)
  * Its own log file (``logging.file`` differs per config)
  * Its own magic number (broker.magic differs per config) so positions
    opened by different pair-bots are clearly separable on the broker side
  * Shared SQLite journal (``data/journal.db``); SQLite handles concurrent
    writers from multiple processes via file locks

Default pair list = config files matching ``config.local.<pair>.yaml`` in the
project root. Override with ``--pairs gbpusd,usdjpy,xauusd`` (or ``--all``).

Run:

    python -m bot.live_multi --all
    python -m bot.live_multi --pairs gbpusd,usdjpy,xauusd
    python -m bot.live_multi --pairs gbpusd --restart-delay 30

Stop with Ctrl+C; the supervisor terminates all children gracefully.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .config import project_root
from .logging_setup import get_logger, setup_logging

log = get_logger(__name__)


@dataclass
class ChildSpec:
    """One supervised LiveBot subprocess."""
    pair: str
    config_path: Path
    proc: Optional[subprocess.Popen] = None
    restarts: int = 0
    last_start: float = 0.0
    crash_log: List[str] = field(default_factory=list)


def discover_configs(root: Path) -> Dict[str, Path]:
    """Return mapping pair -> config path for every ``config.local.<pair>.yaml``."""
    out: Dict[str, Path] = {}
    for p in sorted(root.glob("config.local.*.yaml")):
        # Skip the bare config.local.yaml (no pair token)
        stem = p.stem            # e.g. config.local.gbpusd
        parts = stem.split(".")
        if len(parts) < 3:
            continue
        pair = parts[-1].lower()
        out[pair] = p
    return out


def parse_pair_filter(arg: Optional[str], available: Dict[str, Path]) -> List[str]:
    if not arg or arg.strip().lower() in ("all", "*"):
        return list(available.keys())
    wanted = [p.strip().lower() for p in arg.split(",") if p.strip()]
    missing = [p for p in wanted if p not in available]
    if missing:
        raise SystemExit(
            f"No config found for pair(s): {', '.join(missing)}. "
            f"Available: {', '.join(available) or '(none)'}"
        )
    return wanted


def start_child(spec: ChildSpec) -> None:
    """Spawn ``python -m bot.live --config <path>`` for this pair."""
    cmd = [sys.executable, "-m", "bot.live", "--config", str(spec.config_path)]
    # On Windows we want CTRL_BREAK_EVENT to propagate cleanly; spawn in a
    # new process group so we can target each child individually if needed.
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    spec.proc = subprocess.Popen(
        cmd,
        cwd=str(project_root()),
        stdout=subprocess.DEVNULL,   # the LiveBot writes its own per-pair log file
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    spec.last_start = time.time()
    log.info("[%s] started pid=%s config=%s", spec.pair, spec.proc.pid, spec.config_path.name)


def stop_child(spec: ChildSpec, timeout: float = 10.0) -> None:
    if spec.proc is None or spec.proc.poll() is not None:
        return
    log.info("[%s] stopping pid=%s", spec.pair, spec.proc.pid)
    try:
        if sys.platform == "win32":
            # CTRL_BREAK_EVENT travels through CREATE_NEW_PROCESS_GROUP. The
            # LiveBot installs a SIGBREAK / SIGINT handler so this triggers
            # the same graceful shutdown as Ctrl+C in an interactive terminal.
            spec.proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        else:
            spec.proc.send_signal(signal.SIGTERM)
    except Exception as e:  # noqa: BLE001
        log.warning("[%s] send_signal failed: %s", spec.pair, e)
    try:
        spec.proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        log.warning("[%s] graceful shutdown timed out, killing", spec.pair)
        spec.proc.kill()
        try:
            spec.proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            log.error("[%s] kill also timed out; pid=%s leaked", spec.pair, spec.proc.pid)


def supervise(specs: List[ChildSpec], restart_delay: float, max_restarts_per_hour: int) -> None:
    """Block until SIGINT/Ctrl+C, restarting any child that exits.

    A child that crashes more than ``max_restarts_per_hour`` times is held
    out -- usually that means a config or credential problem we don't want
    to mask with a restart loop. The supervisor stays up so the others keep
    running.
    """
    stopping = {"flag": False}

    def _stop_handler(signum, frame):  # noqa: ARG001
        log.info("Supervisor got signal %s, shutting down children", signum)
        stopping["flag"] = True

    signal.signal(signal.SIGINT, _stop_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _stop_handler)
    if sys.platform == "win32" and hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _stop_handler)  # type: ignore[attr-defined]

    # Initial boot
    for s in specs:
        start_child(s)

    held_out: Dict[str, float] = {}
    try:
        while not stopping["flag"]:
            for s in specs:
                # Skip held-out children
                if s.pair in held_out:
                    continue

                if s.proc is None:
                    continue
                rc = s.proc.poll()
                if rc is None:
                    continue
                # Child exited
                age = time.time() - s.last_start
                log.warning(
                    "[%s] exited after %.0fs with code=%s (restart #%d coming up)",
                    s.pair, age, rc, s.restarts + 1,
                )
                # Backoff if it crashed quickly
                wait = max(restart_delay, 5.0 if age < 30 else 0.0)
                if wait > 0:
                    time.sleep(wait)
                # Crash budget
                if age < 60:
                    s.crash_log.append(time.time())
                    one_hour_ago = time.time() - 3600
                    s.crash_log = [t for t in s.crash_log if t > one_hour_ago]
                    if len(s.crash_log) > max_restarts_per_hour:
                        log.error(
                            "[%s] %d quick crashes in last hour -- holding out. "
                            "Inspect data/bot_%s.log for the cause.",
                            s.pair, len(s.crash_log), s.pair,
                        )
                        held_out[s.pair] = time.time()
                        continue
                s.restarts += 1
                start_child(s)
            time.sleep(1.0)
    finally:
        for s in specs:
            stop_child(s)
        log.info("Supervisor exited")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Axiom Omega multi-pair launcher")
    parser.add_argument(
        "--pairs",
        default=None,
        help="Comma-separated pair tokens (matching config.local.<pair>.yaml). "
             "Use 'all' or omit + --all for every available config.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every available per-pair config",
    )
    parser.add_argument(
        "--restart-delay",
        type=float,
        default=10.0,
        help="Seconds to wait before restarting a crashed child (default 10s)",
    )
    parser.add_argument(
        "--max-crashes-per-hour",
        type=int,
        default=6,
        help="Hold a child out if it crashes more than this many times in 60min",
    )
    args = parser.parse_args(argv)

    # Set up our own logging (the children log to their own files).
    setup_logging({"logging": {"level": "INFO", "file": "data/bot_multi.log"}})

    root = project_root()
    available = discover_configs(root)
    if not available:
        log.error("No config.local.<pair>.yaml files found in %s", root)
        return 1

    pair_arg = args.pairs if args.pairs else ("all" if args.all else None)
    if pair_arg is None:
        # Friendly default: list what's available so the user knows what to do
        pretty = ", ".join(available.keys())
        log.info("Available pair configs: %s", pretty)
        log.info("Pass --all to run all of them, or --pairs gbpusd,usdjpy,...")
        return 0

    pairs = parse_pair_filter(pair_arg, available)
    log.info("Launching %d pair(s): %s", len(pairs), ", ".join(pairs))

    specs = [ChildSpec(pair=p, config_path=available[p]) for p in pairs]
    supervise(
        specs,
        restart_delay=args.restart_delay,
        max_restarts_per_hour=args.max_crashes_per_hour,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
