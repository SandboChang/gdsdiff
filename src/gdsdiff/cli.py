from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._version import __version__
from .api import compare_gds_files
from .config import CompareConfig, parse_allowed_region, parse_allowed_whole_layer
from .inventory import inventory_gds
from .semantics import Backend, CompareStatus, EvidenceEngine, FailurePolicy
from .templates import extract_template, find_template


def build_parser(
    *,
    prog: str = "gdsdiff",
    description: str | None = None,
    default_backend: Backend = Backend.AUTO,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description or "Compare two GDS files with exact CPU verification.",
        epilog=f"Inspect a single layout with '{prog} inspect --help'.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
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
            "Evidence generator to use. For gdsdiff, auto selects gpu-polygon for available GPU backends "
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
    parser.add_argument("--require-gpu", action="store_true", help="Return exit 2 if no GPU comparison ran, even if the CPU fallback completed.")
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
    if argv and argv[0] == "inspect":
        return _main_inspect(argv[1:], prog=f"{prog} inspect")
    if argv and argv[0] == "extract-template":
        return _main_extract_template(argv[1:], prog=f"{prog} extract-template")
    if argv and argv[0] == "find-template":
        return _main_find_template(argv[1:], prog=f"{prog} find-template")

    parser = build_parser(prog=prog, description=description, default_backend=default_backend)
    args = parser.parse_args(argv)
    try:
        _validate_input_paths(args.old_gds, args.new_gds)
    except ValueError as exc:
        if general_output and not args.quiet:
            print("status=unsupported-input evidence_status=unsupported-input differences_found=false")
        print(f"{prog}: {exc}", file=sys.stderr)
        return 2
    try:
        json_report_path, diff_gds_path, markdown_path, exact_diff_markdown_path = _resolve_artifact_paths(args)
        evidence_engine = _resolve_evidence_engine(
            args.evidence_engine,
            args.backend,
            require_gpu=args.require_gpu,
            neutral_fallback=general_output,
        )
        allowed_regions = tuple(parse_allowed_region(value) for value in args.allow_region) + tuple(
            parse_allowed_whole_layer(value) for value in args.allow_whole_layer
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

    exact_fallback = result.report.exact_fallback
    messages: list[str] = []
    if exact_fallback.get("mode") in {
        "cpu-gdstk-oracle-after-gpu-failure", "gpu-record-delta-plus-cpu-oracle-fallback",
    }:
        messages.extend(result.report.warnings)
    elif result.report.backend.get("gpu_work_performed"):
        for failure in exact_fallback.get("failures", ()):
            if failure.get("stage"):
                messages.append(f"{failure['stage']}: {failure['reason']}")
    if result.status in {CompareStatus.INCOMPLETE_EVIDENCE, CompareStatus.INTERNAL_MISMATCH}:
        messages.extend(result.report.errors)
    for diagnostic in exact_fallback.get("gdstk_boolean_diagnostics", ()):
        if diagnostic["outcome"] == "failed" and not any(diagnostic["message"] in msg for msg in messages):
            layer = diagnostic["layer"]
            location = f" on {layer}:{diagnostic['datatype']}" if layer is not None else ""
            messages.append(
                f"{diagnostic['stage']}: rejected gdstk {diagnostic['operation']}{location}: {diagnostic['message']}"
            )
    for message in dict.fromkeys(messages):
        print(f"{prog}: {message}", file=sys.stderr)

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


def _main_inspect(argv: list[str], *, prog: str) -> int:
    parser = argparse.ArgumentParser(prog=prog, description="List GDS units, cells, layers, and stored records without flattening.")
    parser.add_argument("gds", type=Path)
    parser.add_argument("--top", help="Cell to inspect. Defaults to the top-level cell when unambiguous.")
    parser.add_argument("--json", action="store_true", help="Print the inventory as JSON instead of text.")
    args = parser.parse_args(argv)
    try:
        _validate_input_paths(args.gds)
        inventory = inventory_gds(args.gds, top_name=args.top)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"{prog}: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(inventory.to_json(), indent=2))
    else:
        metadata = inventory.metadata
        records = inventory.records
        layers = ", ".join(f"{layer.layer}:{layer.datatype}" for layer in inventory.layers)
        print(f"GDS file: {metadata.path}")
        print(f"User unit: {metadata.unit_m} m; precision: {metadata.precision_m} m")
        print(f"Top-level cells: {', '.join(metadata.top_names) or '<none>'}")
        print(f"Selected cell: {inventory.selected_top}")
        print(f"Cells: {records.cells} reachable / {metadata.cell_count} in file")
        print(f"Geometry layers (layer:datatype): {layers or '<none>'}")
        print(
            f"Stored records: {records.polygons} polygons, {records.paths} paths, "
            f"{records.labels} labels, {records.references} references"
        )
    return 0


def build_extract_template_parser(*, prog: str = "gdsdiff extract-template") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Extract an inspectable normalized template GDS from a source GDS.")
    parser.add_argument("source_gds", type=Path)
    parser.add_argument("--top", dest="top_name", help="Source top cell. Defaults to the GDS top cell when unambiguous.")
    parser.add_argument("--layers", default="all", help="Layer selector such as '1:0,2:0' or 'all'.")
    parser.add_argument("--bbox", required=True, type=_parse_user_bbox, metavar="X0,Y0,X1,Y1", help="Selection bbox in GDS user units.")
    parser.add_argument("--output-gds", required=True, type=Path, help="Standalone template GDS path.")
    parser.add_argument("--metadata", type=Path, help="JSON metadata sidecar. Defaults to OUTPUT_GDS with .json suffix.")
    parser.add_argument("--template-top", default="TEMPLATE", help="Top cell name for the output template GDS.")
    parser.add_argument("--quiet", action="store_true")
    return parser


def build_find_template_parser(*, prog: str = "gdsdiff find-template") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Find exact normalized template matches in a target GDS.")
    parser.add_argument("template_gds", type=Path)
    parser.add_argument("target_gds", type=Path)
    parser.add_argument("--template-top", default="TEMPLATE", help="Template top cell name.")
    parser.add_argument("--target-top", "--top", dest="target_top", help="Target top cell. Defaults to the GDS top cell when unambiguous.")
    parser.add_argument("--layers", default="all", help="Layer selector such as '1:0,2:0' or 'all'.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for matches.json and matches.md.")
    parser.add_argument(
        "--transforms",
        default="translate",
        type=_parse_transform_list,
        help="Comma-separated transform set. Supported values: translate,rot90.",
    )
    parser.add_argument("--allow-mirror", action="store_true", help="Also search mirrored variants of the selected transforms.")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _main_extract_template(argv: list[str], *, prog: str) -> int:
    parser = build_extract_template_parser(prog=prog)
    args = parser.parse_args(argv)
    try:
        metadata_path = args.metadata or args.output_gds.with_suffix(".json")
        result = extract_template(
            args.source_gds,
            output_gds=args.output_gds,
            metadata_path=metadata_path,
            top_name=args.top_name,
            layers=args.layers,
            bbox_user=args.bbox,
            template_top=args.template_top,
        )
    except Exception as exc:
        print(f"{prog}: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        template = result.metadata["template"]
        print(f"template_gds={result.output_gds}")
        print(f"metadata={result.metadata_path}")
        print(f"polygon_count={template['polygon_count']}")
        print(f"signature_sha256={template['signature_sha256']}")
    return 0


def _main_find_template(argv: list[str], *, prog: str) -> int:
    parser = build_find_template_parser(prog=prog)
    args = parser.parse_args(argv)
    try:
        result = find_template(
            args.template_gds,
            args.target_gds,
            output_dir=args.output_dir,
            template_top=args.template_top,
            target_top=args.target_top,
            layers=args.layers,
            transforms=args.transforms,
            allow_mirror=args.allow_mirror,
        )
    except Exception as exc:
        print(f"{prog}: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        summary = result.result["summary"]
        print(f"matches_json={result.matches_json}")
        print(f"matches_markdown={result.matches_markdown}")
        print(f"candidate_count={summary['candidate_count']}")
        print(f"match_count={summary['match_count']}")
        print(f"difference_count={summary['difference_count']}")
    return 0


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


def _validate_input_paths(*paths: Path) -> None:
    for path in paths:
        if not path.is_file():
            raise ValueError(f"input GDS file does not exist or is not a regular file: {path}")


def _parse_user_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must have form X0,Y0,X1,Y1")
    try:
        x0, y0, x1, y1 = (float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox coordinates must be numeric") from exc
    if x0 >= x1 or y0 >= y1:
        raise argparse.ArgumentTypeError("bbox must satisfy X0 < X1 and Y0 < Y1")
    return x0, y0, x1, y1


def _parse_transform_list(value: str) -> tuple[str, ...]:
    transforms = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not transforms:
        raise argparse.ArgumentTypeError("at least one transform is required")
    unknown = set(transforms) - {"translate", "rot90"}
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown transform(s): {', '.join(sorted(unknown))}")
    return transforms


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
    return EvidenceEngine.GPU_POLYGON.value


if __name__ == "__main__":
    raise SystemExit(main())
