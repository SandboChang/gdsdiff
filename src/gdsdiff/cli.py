from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .api import compare_gds_files
from .config import AllowedRegion, CompareConfig, LayerSpec
from .semantics import Backend, DifferenceDirection, EvidenceEngine, FailurePolicy


def build_parser(
    *,
    prog: str = "gdsdiff",
    description: str | None = None,
    default_backend: Backend = Backend.AUTO,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description or "Compare two GDS files with exact CPU verification.",
    )
    parser.add_argument("old_gds", type=Path)
    parser.add_argument("new_gds", type=Path)
    parser.add_argument("--top")
    parser.add_argument("--old-top")
    parser.add_argument("--new-top")
    parser.add_argument("--layers", default="all", help="Layer selector such as '1:0,2:0' or 'all'.")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--backend", choices=[item.value for item in Backend], default=default_backend.value)
    parser.add_argument(
        "--evidence-engine",
        choices=[item.value for item in EvidenceEngine],
        default=EvidenceEngine.AUTO.value,
        help=(
            "Evidence generator to use. For gdsdiff, auto selects gpu-polygon when CUDA is available "
            "and otherwise selects the CPU gdstk oracle."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Write the standard artifact set into this directory: report.json, exact_diff.md, "
            "and the automatically generated diff_vertices.jsonl sidecar."
        ),
    )
    parser.add_argument("--failure-policy", choices=[item.value for item in FailurePolicy], default=FailurePolicy.FAIL_ON_UNEXPECTED.value)
    parser.add_argument("--allow-region", action="append", default=[], metavar="L:D,DIRECTION,X0,Y0,X1,Y1[,HALO]")
    parser.add_argument("--allow-whole-layer", action="append", default=[], metavar="L:D,DIRECTION")
    parser.add_argument("--json-report", type=Path)
    parser.add_argument("--diff-gds", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--exact-diff-markdown", type=Path)
    parser.add_argument("--exact", action="store_true", help="Run full exact oracle instead of CPU tile prefilter.")
    parser.add_argument(
        "--verify-exact-after-fast",
        action="store_true",
        help="For eligible accelerated fast verdicts, run full exact verification afterward and attach polygon-level evidence.",
    )
    parser.add_argument(
        "--audit-cpu-oracle",
        action="store_true",
        help="After a GPU evidence path, run the CPU gdstk oracle as audit evidence when supported.",
    )
    parser.add_argument(
        "--exact-component-backend",
        choices=(
            "none",
            "gdstk-component",
            "rect",
            "rect-fallback",
            "cuda-rect",
            "cuda-rect-fallback",
            "metal-rect",
            "metal-rect-fallback",
        ),
        default="none",
        help="Optional component executor for full exact evidence. 'none' preserves the default gdstk oracle.",
    )
    parser.add_argument(
        "--force-geometry-prefilter",
        action="store_true",
        help="Disable accelerated byte/hierarchy shortcuts and run the extraction plus tile-prefilter geometry path.",
    )
    parser.add_argument(
        "--structural-guard",
        action="store_true",
        help="Prefer the native cold structural count guard before other fast paths.",
    )
    parser.add_argument(
        "--native-geometry-guard",
        action="store_true",
        help="Prefer the custom native GDS geometry summary guard before other fast paths.",
    )
    parser.add_argument(
        "--geometry-cache-dir",
        type=Path,
        help="Optional cache directory for extracted geometry, canonical IDs, and spatial indexes.",
    )
    parser.add_argument("--require-gpu", action="store_true", help="Fail instead of falling back to CPU when no GPU backend is available.")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return _main(
        argv,
        prog="gdsdiff",
        description="General exact GDS diff tool with automatic GPU acceleration and CPU fallback.",
        default_backend=Backend.AUTO,
        general_output=True,
    )


def _main(
    argv: list[str] | None,
    *,
    prog: str,
    description: str,
    default_backend: Backend,
    general_output: bool,
) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser(prog=prog, description=description, default_backend=default_backend)
    args = parser.parse_args(argv)
    try:
        json_report_path, diff_gds_path, markdown_path, exact_diff_markdown_path = _resolve_artifact_paths(args)
        evidence_engine = _resolve_evidence_engine(
            args.evidence_engine,
            args.backend,
            require_gpu=args.require_gpu,
            neutral_fallback=general_output,
        )
        allowed_regions = tuple(_parse_allowed_region(value) for value in args.allow_region) + tuple(
            _parse_allowed_whole_layer(value) for value in args.allow_whole_layer
        )
        config = CompareConfig(
            old_path=args.old_gds,
            new_path=args.new_gds,
            top=args.top,
            old_top=args.old_top,
            new_top=args.new_top,
            layers=args.layers,
            backend=args.backend,
            evidence_engine=evidence_engine,
            tile_size=args.tile_size,
            allowed_regions=allowed_regions,
            failure_policy=args.failure_policy,
            geometry_cache_dir=args.geometry_cache_dir,
            require_gpu=args.require_gpu,
        )
        result = compare_gds_files(
            config,
            json_report_path=json_report_path,
            diff_gds_path=diff_gds_path,
            markdown_path=markdown_path,
            exact_diff_markdown_path=exact_diff_markdown_path,
            exact=args.exact,
            verify_exact_after_fast=args.verify_exact_after_fast,
            audit_cpu_oracle=args.audit_cpu_oracle,
            use_fast_shortcuts=not args.force_geometry_prefilter,
            use_structural_guard=args.structural_guard,
            use_native_geometry_guard=args.native_geometry_guard,
            exact_component_backend=args.exact_component_backend,
            command=(prog, *argv),
        )
    except Exception as exc:
        print(f"{prog}: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        report = result.report.to_json_dict()
        print(
            "status={status} evidence_status={evidence_status} differences_found={differences_found}".format(
                status=report["status"],
                evidence_status=report["evidence_status"],
                differences_found=str(report["differences_found"]).lower(),
            )
        )
        if report.get("coverage_changed") is not None or report.get("inventory_changed") is not None:
            print(
                "coverage_changed={coverage_changed} inventory_changed={inventory_changed}".format(
                    coverage_changed=str(report.get("coverage_changed")).lower(),
                    inventory_changed=str(report.get("inventory_changed")).lower(),
                )
            )
        if result.artifacts:
            for name, path in sorted(result.artifacts.items()):
                print(f"{name}={path}")
    return result.exit_code


def _resolve_artifact_paths(args: argparse.Namespace) -> tuple[Path | None, Path | None, Path | None, Path | None]:
    json_report_path = args.json_report
    diff_gds_path = args.diff_gds
    markdown_path = args.markdown
    exact_diff_markdown_path = args.exact_diff_markdown
    if args.output_dir is not None:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        json_report_path = json_report_path or output_dir / "report.json"
        exact_diff_markdown_path = exact_diff_markdown_path or output_dir / "exact_diff.md"
    return json_report_path, diff_gds_path, markdown_path, exact_diff_markdown_path


def _resolve_evidence_engine(
    evidence_engine: str,
    backend: str,
    *,
    require_gpu: bool = False,
    neutral_fallback: bool = False,
) -> str:
    if EvidenceEngine.coerce(evidence_engine) != EvidenceEngine.AUTO:
        return evidence_engine
    if not neutral_fallback:
        return evidence_engine
    requested_backend = Backend.coerce(backend)
    if requested_backend == Backend.METAL:
        return EvidenceEngine.GDSTK_ORACLE.value
    return EvidenceEngine.GPU_POLYGON.value


def _parse_allowed_region(value: str) -> AllowedRegion:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) not in {6, 7}:
        raise ValueError("allowed region must have form L:D,DIRECTION,X0,Y0,X1,Y1[,HALO]")
    layer = LayerSpec.parse(parts[0])
    direction = DifferenceDirection.coerce(parts[1])
    x0, y0, x1, y1 = (int(part) for part in parts[2:6])
    halo = int(parts[6]) if len(parts) == 7 else 0
    return AllowedRegion(layers=(layer,), direction=direction, bbox=(x0, y0, x1, y1), halo_dbu=halo)


def _parse_allowed_whole_layer(value: str) -> AllowedRegion:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError("whole-layer allowance must have form L:D,DIRECTION")
    return AllowedRegion(layers=(LayerSpec.parse(parts[0]),), direction=DifferenceDirection.coerce(parts[1]), whole_layer=True)


if __name__ == "__main__":
    raise SystemExit(main())
