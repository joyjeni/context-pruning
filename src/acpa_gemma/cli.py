"""Command-line entrypoint for Kaggle and local runs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import List

from acpa_gemma.config import load_config
from acpa_gemma.gemma_client import GemmaGenerationError
from acpa_gemma.pipeline import PipelineInputError, TrustSafetyPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Gemma 4 Trust & Safety analysis with ACPA pruning."
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Path to app TOML config. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--secrets",
        action="append",
        default=[],
        help="Path to secrets TOML config. Can be supplied multiple times.",
    )
    parser.add_argument("--input", help="Agentic Eval dataset path.")
    parser.add_argument("--output", help="Output JSONL path.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Limit records for quick Kaggle notebook iterations.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use deterministic local responses without calling Gemma.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    config = load_config(
        config_paths=[Path(path) for path in args.config] if args.config else None,
        secret_paths=[Path(path) for path in args.secrets] if args.secrets else None,
    )
    pipeline = TrustSafetyPipeline(config=config, dry_run=args.dry_run)
    output_path = args.output or config.output.path
    try:
        outputs = pipeline.run_to_file(
            input_dir=args.input,
            output_path=args.output,
            sample_size=args.sample_size,
        )
    except (PipelineInputError, GemmaGenerationError) as exc:
        print(
            json.dumps(
                {
                    "error": str(exc),
                    "output_path": output_path,
                    "model": config.gemma.model,
                    "loaded_config_files": config.loaded_files,
                    "dry_run": args.dry_run,
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            {
                "records_processed": len(outputs),
                "output_path": output_path,
                "model": config.gemma.model,
                "loaded_config_files": config.loaded_files,
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
