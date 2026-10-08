#!/usr/bin/env python
"""Prepare/freeze/score future-data research; no broker, scheduler or notifications.

A scoring command refuses to write performance before a full declared checkpoint.
A failed readiness command is not an initialized OOS experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trend_following.research_oos_data import (  # noqa:E402
    normalize_yahoo_frames,
    read_calendar,
    write_input_package,
)
from trend_following.research_oos_protocol import (  # noqa:E402
    assess_readiness,
    freeze_oos_study,
    load_oos_config,
    now_utc,
    score_oos_study,
    verify_oos_freeze,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/qqq_oos_v2.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)
    acquire = sub.add_parser(
        "acquire", help="download completed Yahoo inputs to NEW immutable package"
    )
    acquire.add_argument("--output", type=Path, required=True)
    acquire.add_argument(
        "--distribution-calendar",
        type=Path,
        help="separate verified future ex/pay calendar; never edit frozen historical CSV",
    )
    acquire.add_argument(
        "--fallback-30min-dir",
        type=Path,
        help="licensed AV cache with verified start-labeled adjusted 30min bars",
    )
    acquire.add_argument(
        "--fallback-daily-dir",
        type=Path,
        help="matching AV raw/adjusted daily quotes for physical-unit restoration",
    )
    acquire.add_argument(
        "--fallback-15min-dir",
        type=Path,
        help="preserved 15min cache to verify nested close/start labeling",
    )
    prepare = sub.add_parser(
        "prepare", help="normalize previously downloaded frames, preserving raw inputs"
    )
    prepare.add_argument("--source-dir", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--distribution-calendar", type=Path)
    prepare.add_argument("--fallback-30min-dir", type=Path)
    prepare.add_argument("--fallback-daily-dir", type=Path)
    prepare.add_argument("--fallback-15min-dir", type=Path)
    check = sub.add_parser(
        "check", help="audit coverage/calibration without declaring a study start"
    )
    check.add_argument("--warmup", type=Path, required=True)
    check.add_argument("--output", type=Path, help="NEW readiness report; will not overwrite")
    freeze = sub.add_parser(
        "freeze", help="final freeze AFTER successful coverage/calibration tests"
    )
    freeze.add_argument("--warmup", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--attest-ready", action="store_true")
    verify = sub.add_parser("verify", help="recheck exact frozen code, parameters and dependencies")
    verify.add_argument("--manifest", type=Path, required=True)
    score = sub.add_parser("score", help="score ONLY a fully elapsed 12/24-month checkpoint")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--future-package", type=Path, action="append", required=True)
    score.add_argument("--months", type=int, choices=[12, 24], required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if root != Path(__file__).resolve().parents[1]:
        parser.error(
            "--root must match the executing checkout; copied-tree hashes cannot authorize different loaded code"
        )

    def resolve(path):
        return path if path.is_absolute() else root / path

    config_path = resolve(args.config)
    if args.command in {"prepare", "acquire"}:
        import pandas as pd

        config = load_oos_config(config_path)
        cal = read_calendar(resolve(Path(config["calendar"])))
        ref_path = resolve(args.distribution_calendar or Path(config["distribution_calendar"]))
        reference = pd.read_csv(ref_path)
        hourly, daily, raw, acquisition = {}, {}, {}, {}
        if args.command == "acquire":
            import tempfile

            import yfinance as yf

            source = Path(tempfile.mkdtemp(prefix="qqq_oos_source_"))
            for asset in ["QQQ", "TQQQ"]:
                ticker = yf.Ticker(asset)
                common = dict(
                    period="2y",
                    auto_adjust=False,
                    back_adjust=False,
                    actions=True,
                    prepost=False,
                    repair=False,
                )
                hourly[asset] = ticker.history(interval="60m", **common)
                daily[asset] = ticker.history(interval="1d", **common)
                observed = now_utc()
                hourly[asset].to_parquet(source / f"{asset}_hourly.parquet")
                daily[asset].to_parquet(source / f"{asset}_daily.parquet")
                info = {
                    "observed_at_utc": observed.isoformat(),
                    "provider": "Yahoo Finance via yfinance",
                    "yfinance_version": yf.__version__,
                    "request": {**common, "intervals": ["60m", "1d"]},
                }
                (source / f"{asset}_metadata.json").write_text(json.dumps(info, indent=2) + "\n")
        else:
            source = args.source_dir.resolve()
            for asset in ["QQQ", "TQQQ"]:
                hourly[asset] = pd.read_parquet(source / f"{asset}_hourly.parquet")
                daily[asset] = pd.read_parquet(source / f"{asset}_daily.parquet")
        for asset in ["QQQ", "TQQQ"]:
            meta_path = source / f"{asset}_metadata.json"
            if not meta_path.is_file():
                raise ValueError("original acquisition timestamp/request metadata required")
            acquisition[asset] = json.loads(meta_path.read_text())
            for suffix in ["hourly.parquet", "daily.parquet", "metadata.json"]:
                raw[f"{asset}_{suffix}"] = source / f"{asset}_{suffix}"
        fallback_30, fallback_daily, fallback_15 = None, None, None
        if (args.fallback_30min_dir is None) != (args.fallback_daily_dir is None):
            raise ValueError("both fallback directories are required together")
        if args.fallback_30min_dir is not None:
            if args.fallback_15min_dir is None:
                raise ValueError("fallback requires 15min timing-proof directory too")
            fallback_30, fallback_daily, fallback_15 = {}, {}, {}
            for asset in ["QQQ", "TQQQ"]:
                bar_file = args.fallback_30min_dir / f"{asset}.parquet"
                day_file = args.fallback_daily_dir / f"{asset}.parquet"
                fallback_30[asset] = pd.read_parquet(bar_file)
                fallback_daily[asset] = pd.read_parquet(day_file)
                raw[f"{asset}_av30.parquet"] = bar_file
                raw[f"{asset}_avdaily.parquet"] = day_file
                proof_file = args.fallback_15min_dir / f"{asset}.parquet"
                fallback_15[asset] = pd.read_parquet(proof_file)
                raw[f"{asset}_av15.parquet"] = proof_file
        now = now_utc()
        prices, actions, basis = normalize_yahoo_frames(
            hourly,
            daily,
            cal,
            reference,
            observed_at=now,
            fallback_30min=fallback_30,
            fallback_daily=fallback_daily,
            fallback_15min=fallback_15,
        )
        raw["distribution_calendar.csv"] = ref_path
        metadata = write_input_package(
            resolve(args.output),
            prices,
            actions,
            cal,
            observed_at=now,
            provenance={
                **basis,
                "source_acquisition": acquisition,
                "adapter": "qqq_oos_v2_Yahoo_AV_full_session_v1",
                "fallback_supplied": fallback_30 is not None,
                "fallback_acquisition": "licensed_local_cache_read_now_not_point_in_time_download",
            },
            raw_files=raw,
        )
        result = {
            "package": str(resolve(args.output)),
            "quality": metadata["quality"],
            "status": "input_package_not_a_study_freeze",
        }
    elif args.command == "check":
        result = assess_readiness(root, config_path, resolve(args.warmup))
        if args.output:
            with resolve(args.output).open("x") as stream:
                stream.write(json.dumps(result, indent=2) + "\n")
    elif args.command == "freeze":
        result = freeze_oos_study(
            root,
            config_path,
            resolve(args.warmup),
            resolve(args.output),
            attest_ready=args.attest_ready,
        )
    elif args.command == "verify":
        result = verify_oos_freeze(root, resolve(args.manifest))
    else:
        result = score_oos_study(
            root,
            resolve(args.manifest),
            [resolve(p) for p in args.future_package],
            args.months,
            resolve(args.output),
        )
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.command == "check" and not result["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
