#!/usr/bin/env python3
"""Read-only v5 audit/strict prepare; downloads only explicitly named raw months.

Examples (no backtest/performance outputs are read or generated)::

  python scripts/prepare_qqq_hourly_v5_data.py audit-legacy
  python scripts/prepare_qqq_hourly_v5_data.py download-month 2004-11 --source-env /path/to/.env
  python scripts/prepare_qqq_hourly_v5_data.py prepare --raw-30min /path/raw.parquet \\
      --metadata /path/verified_metadata.json --daily /path/daily.parquet \\
      --start 1999-11-01 --bar-start 2000-01-03 --end 2026-06-01 \\
      --as-of 2026-10-08T15:00:00Z --output /path/new_packet

Downloading CSV does NOT establish timestamp semantics; verified metadata is
required separately. Existing adjusted caches cannot pass the prepare command.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trend_following.research_hourly_data_v5 import (  # noqa: E402
    DataContractError,
    SourceMetadata,
    audit_alpha_only_cache,
    audit_source_coverage,
    build_xnys_calendar,
    download_alpha_vantage_raw_month,
    normalize_daily_actions,
    normalize_monitor_subbars,
    normalize_raw_subbars,
    prepare_input_packet,
    prepare_monitor_packet,
    sha256_file,
    utc,
    validate_cash_observations,
)

# A clone must not depend on the original researcher's machine. External cache
# locations are selected explicitly with --source-root.
ORIGINAL_SOURCE = ROOT


def read_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def audit_legacy(args: argparse.Namespace) -> int:
    """Inspect caches without mutation or treating adjusted values as raw."""
    data_root = args.source_root / "data/raw"
    daily_path = data_root / "alpha_vantage_daily_adjusted/QQQ.parquet"
    original_daily = pd.read_parquet(daily_path)
    start = pd.Timestamp(original_daily.date.iloc[0]).strftime("%Y-%m-%d")
    end = pd.Timestamp(original_daily.date.iloc[-1]).strftime("%Y-%m-%d")
    calendar = build_xnys_calendar(start, end)
    daily = normalize_daily_actions(original_daily, calendar)
    report = {
        "kind": "v5_read_only_input_readiness",
        "performance_generated": False,
        "historical_ready": False,
        "reason": "no verified raw intraday source; legacy downloader requested adjusted=true",
        "daily": {
            "path": str(daily_path),
            "sha256": sha256_file(daily_path),
            "rows": len(daily),
            "start_session": start,
            "end_session": end,
            "price_field": "close",
            "price_basis": "raw_as_traded",
            "schema": "close,dividend_amount,split_coefficient",
            "coverage_complete": True,
        },
        "calendar_provenance": calendar.attrs["calendar_provenance"],
        "sources": {},
    }
    if args.output_dir.exists():
        raise FileExistsError(f"immutable audit output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    calendar.to_csv(args.output_dir / "calendar.csv", index=False)
    for minutes in (15, 30, 60):
        path = data_root / f"alpha_vantage_{minutes}min/QQQ.parquet"
        frame = pd.read_parquet(path)
        source_start = pd.Timestamp(frame.date.iloc[0]).strftime("%Y-%m-%d")
        source_calendar = calendar[calendar.session.ge(source_start)].reset_index(drop=True)
        meta = SourceMetadata(
            "Alpha Vantage legacy adjusted cache",
            "QQQ",
            minutes,
            "start" if minutes in (15, 30) else "wall_hour_not_canonical",
            "America/New_York",
            "split_dividend_adjusted",
            True,
            pd.Timestamp.now(tz="UTC").isoformat(),
            "existing data_download.py explicitly requests adjusted=true",
            (str(path),),
        )
        gaps = audit_source_coverage(frame, source_calendar, meta)
        gaps.to_csv(args.output_dir / f"legacy_{minutes}min_gap_report.csv", index=False)
        report["sources"][str(minutes)] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(frame),
            "first_timestamp": str(frame.date.iloc[0]),
            "last_timestamp": str(frame.date.iloc[-1]),
            "adjusted": True,
            "price_basis": "split_dividend_adjusted",
            "normalization_allowed": False,
            "issues": gaps.groupby("issue").size().to_dict(),
            "missing_by_session": gaps[gaps.issue.eq("missing_subbar")]
            .groupby("session")
            .size()
            .to_dict(),
            "gap_report": f"legacy_{minutes}min_gap_report.csv",
            "note": "60m wall-hour labels are not relabelled as canonical 09:30 hours",
        }
    cash_path = data_root / "research_rates/20261002_research_v1/DTB3_available.csv"
    if cash_path.is_file():
        cash = pd.read_csv(cash_path)
        now = pd.Timestamp.now(tz="UTC")
        validated_cash = validate_cash_observations(
            cash, as_of=now, first_bar_start=calendar.market_open.iloc[0]
        )
        report["cash_observations"] = {
            "path": str(cash_path),
            "sha256": sha256_file(cash_path),
            "rows": len(validated_cash),
            "columns": list(cash.columns),
            "first_available_at": validated_cash.available_at.iloc[0].isoformat(),
            "last_available_at": validated_cash.available_at.iloc[-1].isoformat(),
            "schema_valid": True,
            "annual_rate_units": "annual_decimal",
            "duplicate_availability_timestamps": int(
                validated_cash.available_at.duplicated().sum()
            ),
            "availability_tie_policy": "latest_observation_date_at_same_available_at",
            "availability": "explicit_source_assumption_latest_vintage_not_point_in_time",
        }
    else:
        report["cash_observations"] = {
            "path": str(cash_path),
            "schema_valid": False,
            "reason": "missing_timestamped_cash_observations",
        }
    probes = ROOT / "data/raw/v5_alpha_vantage_raw_30min/QQQ"
    report["raw_source_availability_probes"] = []
    if probes.is_dir():
        for path in sorted(probes.glob("*.metadata.json")):
            probe = json.loads(path.read_text())
            report["raw_source_availability_probes"].append(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "request": probe.get("request", {}),
                    "response_status": probe.get("response_status"),
                    "http_status": probe.get("http_status"),
                    "observed_at": probe.get("observed_at"),
                    "note": "provider denied premium endpoint; no subscription or data fallback",
                }
            )
    (args.output_dir / "readiness.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "historical_ready": False,
                "performance_generated": False,
                "audit_directory": str(args.output_dir.resolve()),
                "reason": report["reason"],
            },
            indent=2,
        )
    )
    return 2


def audit_alpha_only(args: argparse.Namespace) -> int:
    """Write diagnostics only; never reconstructed quotes or a raw packet."""
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(
            f"immutable Alpha-only audit output already exists: {args.output_dir}"
        )
    report, gaps, calendar = audit_alpha_only_cache(
        args.source_root,
        bar_start_session=args.bar_start,
        close_rtol=args.close_rtol,
        close_atol=args.close_atol,
    )
    args.output_dir.mkdir(parents=True)
    gaps.to_csv(args.output_dir / "alpha_30min_gap_report.csv", index=False)
    calendar.to_csv(args.output_dir / "calendar.csv", index=False)
    (args.output_dir / "readiness.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "audit_directory": str(args.output_dir.resolve()),
                "cutoff_session": report["cutoff_session"],
                "provider_policy": report["provider_policy"],
                "historical_ready": False,
                "performance_generated": False,
                "prepared_raw_packet_generated": False,
                "required_missing_expected_timestamps": report["missing_expected_timestamps"],
            },
            indent=2,
        )
    )
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser(
        "audit-legacy", help="audit existing caches; never normalize adjusted values as raw"
    )
    audit.add_argument("--source-root", type=Path, default=ORIGINAL_SOURCE)
    audit.add_argument(
        "--output-dir", type=Path, default=ROOT / ".cache/qqq_daily_hourly_v5/data_readiness"
    )
    alpha_only = commands.add_parser(
        "audit-alpha-only",
        help="read-only existing Alpha30/daily feasibility, stop at common terminal date",
    )
    alpha_only.add_argument("--source-root", type=Path, default=ORIGINAL_SOURCE)
    alpha_only.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / ".cache/qqq_daily_halfhour_v5_alpha_only/data_readiness_20261008",
    )
    alpha_only.add_argument("--bar-start", default="2001-01-02")
    alpha_only.add_argument("--close-rtol", type=float, default=0.001)
    alpha_only.add_argument("--close-atol", type=float, default=0.02)
    download = commands.add_parser(
        "download-month", help="explicit one-month raw request; immutable isolated cache"
    )
    download.add_argument("month")
    download.add_argument("--source-env", type=Path, required=True)
    download.add_argument(
        "--cache-dir", type=Path, default=ROOT / "data/raw/v5_alpha_vantage_raw_30min"
    )
    prepare = commands.add_parser(
        "prepare", help="prepare only verified RAW30m complete historical inputs"
    )
    for name in ("raw-30min", "metadata", "daily", "output"):
        prepare.add_argument(f"--{name}", type=Path, required=True)
    for name in ("start", "bar-start", "end", "as-of"):
        prepare.add_argument(f"--{name}", required=True)
    prepare.add_argument("--cash-observations", type=Path)
    prepare.add_argument("--close-rtol", type=float, default=0.001)
    prepare.add_argument("--close-atol", type=float, default=0.02)
    prepare.add_argument("--sampling-interval-minutes", type=int, choices=(30, 60), default=30)
    monitor = commands.add_parser(
        "prepare-monitor",
        help="prepare a distinct completed-bar monitor packet; no today's daily close",
    )
    for name in ("raw-30min", "metadata", "daily", "output"):
        monitor.add_argument(f"--{name}", type=Path, required=True)
    monitor.add_argument("--start", default="1999-11-01")
    for name in ("bar-start", "as-of"):
        monitor.add_argument(f"--{name}", required=True)
    monitor.add_argument("--cash-observations", type=Path)
    monitor.add_argument("--close-rtol", type=float, default=0.001)
    monitor.add_argument("--close-atol", type=float, default=0.02)
    monitor.add_argument("--sampling-interval-minutes", type=int, choices=(30, 60), default=30)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "audit-legacy":
            return audit_legacy(args)
        if args.command == "audit-alpha-only":
            return audit_alpha_only(args)
        if args.command == "download-month":
            from dotenv import dotenv_values

            key = dotenv_values(args.source_env).get("ALPHA_VANTAGE_API_KEY")
            if not key:
                raise DataContractError(
                    "authorized Alpha Vantage credential not present in specified source .env"
                )
            receipt = download_alpha_vantage_raw_month(args.month, args.cache_dir, api_key=key)
            print(json.dumps(receipt, indent=2))
            return 0
        metadata_dict = json.loads(args.metadata.read_text())
        daily_source_metadata = metadata_dict.pop("daily_source_metadata", None)
        metadata_dict["source_files"] = tuple(
            dict.fromkeys(
                (
                    *metadata_dict.get("source_files", ()),
                    str(args.raw_30min.resolve()),
                    str(args.daily.resolve()),
                    str(args.metadata.resolve()),
                )
            )
        )
        meta = SourceMetadata(**metadata_dict)
        monitoring = args.command == "prepare-monitor"
        end = (
            utc(args.as_of, "as_of").tz_convert("America/New_York").strftime("%Y-%m-%d")
            if monitoring
            else args.end
        )
        cal = build_xnys_calendar(args.start, end)
        original_daily = read_frame(args.daily)
        daily_cal = cal.loc[cal.session.lt(end)].reset_index(drop=True) if monitoring else cal
        daily = normalize_daily_actions(original_daily, daily_cal)
        if monitoring and len(daily) != len(original_daily):
            raise DataContractError(
                "monitor daily input must contain only complete prior calendar sessions; today's official close is prohibited"
            )
        bar_cal = cal[cal.session.ge(args.bar_start)].reset_index(drop=True)
        normalize = normalize_monitor_subbars if monitoring else normalize_raw_subbars
        bars = normalize(
            read_frame(args.raw_30min),
            bar_cal,
            meta,
            as_of=args.as_of,
            sampling_interval_minutes=args.sampling_interval_minutes,
        )
        cash = None
        if args.cash_observations:
            cash = read_frame(args.cash_observations)
            cash = cash[
                pd.to_datetime(cash.available_at, utc=True).le(pd.Timestamp(args.as_of))
            ].reset_index(drop=True)
        prepare_packet = prepare_monitor_packet if monitoring else prepare_input_packet
        manifest = prepare_packet(
            args.output,
            daily,
            bars,
            cal,
            meta,
            as_of=args.as_of,
            bar_start_session=args.bar_start,
            close_rtol=args.close_rtol,
            close_atol=args.close_atol,
            cash_observations=cash,
            cash_source_file=args.cash_observations,
            daily_source_metadata=daily_source_metadata,
            sampling_interval_minutes=args.sampling_interval_minutes,
        )
        print(
            json.dumps({"packet": str(args.output.resolve()), "audit": manifest["audit"]}, indent=2)
        )
        return 0
    except (
        DataContractError,
        FileExistsError,
        OSError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(f"STOP: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
