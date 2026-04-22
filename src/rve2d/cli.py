from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rve2d.config import load_config
from rve2d.study import run_batch_study
from rve2d.workflow import build_and_solve_rve, build_rve, solve_with_ferrite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rve2d",
        description="2D ply-scale RVE generator and mesher.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate-config",
        help="Validate a YAML or JSON config.",
    )
    validate_parser.add_argument("config_path", type=Path)

    build_parser = subparsers.add_parser(
        "build",
        help="Generate geometry, mesh it, and export results.",
    )
    build_parser.add_argument("config_path", type=Path)
    build_parser.add_argument("--output-dir", type=Path, default=None)
    build_parser.add_argument("--basename", type=str, default=None)

    solve_parser = subparsers.add_parser(
        "solve-ferrite",
        help="Run the Ferrite.jl homogenization solve on an existing .msh mesh.",
    )
    solve_parser.add_argument("config_path", type=Path)
    solve_parser.add_argument("mesh_path", type=Path)
    solve_parser.add_argument("--output-dir", type=Path, required=True)

    build_solve_parser = subparsers.add_parser(
        "build-and-solve",
        help="Build the RVE and run the Ferrite.jl solve pipeline.",
    )
    build_solve_parser.add_argument("config_path", type=Path)
    build_solve_parser.add_argument("--output-dir", type=Path, default=None)
    build_solve_parser.add_argument("--basename", type=str, default=None)

    batch_parser = subparsers.add_parser(
        "batch-study",
        help="Run build-and-solve across multiple configs and write a study summary CSV.",
    )
    batch_parser.add_argument("config_paths", type=Path, nargs="+")
    batch_parser.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args(argv)

    if args.command == "batch-study":
        case_results, summary_csv = run_batch_study(args.config_paths, args.output_dir)
        batch_payload: dict[str, Any] = {
            "output_directory": str(args.output_dir),
            "summary_csv": str(summary_csv),
            "cases": [
                {
                    "case_name": case.case_name,
                    "config_path": str(case.config_path),
                    "output_directory": str(case.result.output_directory),
                    "summary_path": (
                        str(case.result.ferrite_result.summary_path)
                        if case.result.ferrite_result is not None
                        else None
                    ),
                    "vtk_path": (
                        str(case.result.ferrite_result.vtk_path)
                        if case.result.ferrite_result is not None
                        and case.result.ferrite_result.vtk_path is not None
                        else None
                    ),
                }
                for case in case_results
            ],
        }
        print(json.dumps(batch_payload, indent=2))
        return 0

    config = load_config(args.config_path)

    if args.command == "validate-config":
        print(json.dumps(config.to_dict(), indent=2))
        return 0

    if args.command == "solve-ferrite":
        ferrite_result = solve_with_ferrite(config, args.mesh_path, args.output_dir)
        solve_payload: dict[str, Any] = {
            "summary_path": str(ferrite_result.summary_path),
            "stdout_path": str(ferrite_result.stdout_path),
            "vtk_path": str(ferrite_result.vtk_path) if ferrite_result.vtk_path else None,
            "homogenized_stiffness": ferrite_result.homogenized_stiffness,
            "engineering_constants": ferrite_result.engineering_constants,
            "response_files": [str(path) for path in ferrite_result.response_files],
        }
        print(json.dumps(solve_payload, indent=2))
        return 0

    builder = build_rve if args.command == "build" else build_and_solve_rve
    result = builder(
        config,
        output_dir=str(args.output_dir) if args.output_dir else None,
        basename=args.basename,
    )
    payload: dict[str, Any] = {
        "output_directory": str(result.output_directory),
        "mesh_files": [str(path) for path in result.mesh_files],
        "metadata_files": [str(path) for path in result.metadata_files],
        "quality_report": asdict(result.quality_report),
        "geometry_metadata": result.geometry_metadata,
    }
    if result.ferrite_result is not None:
        payload["ferrite"] = {
            "summary_path": str(result.ferrite_result.summary_path),
            "stdout_path": str(result.ferrite_result.stdout_path),
            "vtk_path": (
                str(result.ferrite_result.vtk_path) if result.ferrite_result.vtk_path else None
            ),
            "homogenized_stiffness": result.ferrite_result.homogenized_stiffness,
            "engineering_constants": result.ferrite_result.engineering_constants,
            "response_files": [str(path) for path in result.ferrite_result.response_files],
        }
    print(json.dumps(payload, indent=2))
    return 0
