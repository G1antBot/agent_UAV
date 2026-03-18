"""
agent_UAV – main entry point
============================
Run a closed-loop UAV mission from the command line::

    python -m src.main "起飞到10米，向北飞行100米，检测是否有人，返航"

Or with a config file::

    python -m src.main --config configs/config.yaml "your task here"

Environment variables
---------------------
VOLC_ACCESSKEY   – Volcano Engine API key
VOLC_ENDPOINT_ID – Ark inference endpoint ID
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="agent_UAV: natural language → UAV control closed-loop simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "task",
        nargs="?",
        default="起飞到5米高度，悬停3秒，然后降落",
        help="Natural language mission task (default: simple takeoff & land demo)",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to configs/config.yaml (optional)",
    )
    parser.add_argument(
        "--simulator",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the in-process UAV simulator (default: True)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()  # Load .env file if present
    args = _parse_args(argv)
    _setup_logging(args.verbose)

    # Import here so logging is configured first
    from src.orchestrator import Orchestrator

    config_path = args.config
    if config_path is None:
        # Auto-detect config if running from the repo root
        candidate = os.path.join(
            os.path.dirname(__file__), "..", "configs", "config.yaml"
        )
        if os.path.exists(candidate):
            config_path = candidate

    with Orchestrator(
        config_path=config_path,
        use_simulator=args.simulator,
    ) as orchestrator:
        report = orchestrator.run_mission(args.task)

    print("\n=== Mission Report ===")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
