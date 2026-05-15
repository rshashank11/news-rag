#!/usr/bin/env python3
"""Bulk upload .env entries to Hugging Face Space secrets/variables.

Usage examples:
  # Dry run first (recommended)
  uv run python scripts/upload_space_env.py \
    --space-id your-username/your-space-name \
    --dry-run

  # Upload everything as secrets
  uv run python scripts/upload_space_env.py \
    --space-id your-username/your-space-name

  # Upload selected keys as variables (non-secret), rest as secrets
  uv run python scripts/upload_space_env.py \
    --space-id your-username/your-space-name \
    --variable-keys DEFAULT_NEWS_SOURCE,RERANK_MODE

Requirements:
  - HF write token must be present in env (default: HF_TOKEN)
  - `huggingface_hub` installed (you already added this via uv)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


RESERVED_PREFIXES = ("#",)


def parse_dotenv(env_path: Path) -> dict[str, str]:
    data: dict[str, str] = {}

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith(RESERVED_PREFIXES):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        # Remove surrounding single/double quotes.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        data[key] = value

    return data


def parse_csv_keys(value: str | None) -> set[str]:
    if not value:
        return set()

    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload .env entries to Hugging Face Space secrets/variables.",
    )
    parser.add_argument(
        "--space-id",
        required=True,
        help="Hugging Face Space repo ID in the form <username>/<space-name>.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (default: .env).",
    )
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable name that stores HF token (default: HF_TOKEN).",
    )
    parser.add_argument(
        "--variable-keys",
        default="",
        help=(
            "Comma-separated keys to upload as Space variables (non-secret). "
            "All other keys are uploaded as secrets."
        ),
    )
    parser.add_argument(
        "--only-keys",
        default="",
        help="Optional comma-separated allowlist of keys to upload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without uploading anything.",
    )

    args = parser.parse_args()

    env_path = Path(args.env_file)
    if not env_path.exists():
        raise FileNotFoundError(f".env file not found: {env_path}")

    token = os.environ.get(args.token_env, "").strip()
    if not token:
        raise RuntimeError(
            f"Missing token in environment variable {args.token_env}. "
            f"Set it first, e.g. export {args.token_env}=hf_xxx"
        )

    variable_keys = parse_csv_keys(args.variable_keys)
    only_keys = parse_csv_keys(args.only_keys)

    env_data = parse_dotenv(env_path)
    if only_keys:
        env_data = {k: v for k, v in env_data.items() if k in only_keys}

    if not env_data:
        print("No keys found to upload.")
        return 0

    api = HfApi(token=token)

    print(f"Space: {args.space_id}")
    print(f"Env file: {env_path}")
    print(f"Total keys to process: {len(env_data)}")

    for key, value in env_data.items():
        upload_as_variable = key in variable_keys
        target = "variable" if upload_as_variable else "secret"

        if args.dry_run:
            print(f"[DRY-RUN] {key} -> {target}")
            continue

        if upload_as_variable:
            api.add_space_variable(
                repo_id=args.space_id,
                key=key,
                value=value,
            )
        else:
            api.add_space_secret(
                repo_id=args.space_id,
                key=key,
                value=value,
            )

        print(f"Uploaded {key} as {target}")

    if args.dry_run:
        print("Dry run complete. No changes were sent to Hugging Face.")
    else:
        print("Upload complete.")
        print("If your Space is already running, restart it to pick up new values.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
