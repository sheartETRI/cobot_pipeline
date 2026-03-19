# run_demo.py (updated)
from __future__ import annotations

import json
from pathlib import Path
import argparse
import sys
from typing import Tuple, List

from pydantic import ValidationError

from executor import run_ir
from ir_models import GenericCobotIR, VerificationResult
from world_model import WorldModel


def load_ir(path: str) -> GenericCobotIR:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return GenericCobotIR.model_validate(data)
    except FileNotFoundError:
        print(f"[ERROR] IR file not found: {path}")
        raise
    except ValidationError as e:
        print(f"[ERROR] IR validation failed for {path}: {e}")
        raise


def load_world(path: str) -> WorldModel:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return WorldModel.model_validate(data)
    except FileNotFoundError:
        print(f"[ERROR] world file not found: {path}")
        raise
    except ValidationError as e:
        print(f"[ERROR] world validation failed for {path}: {e}")
        raise


def resolve_world_path(ir_path: str) -> Path | None:
    ir_file = Path(ir_path)
    if ir_file.suffix != ".json":
        return None
    candidate = ir_file.with_name(f"{ir_file.stem}.world.json")
    return candidate if candidate.exists() else None


def validate_ir_world_consistency(ir: GenericCobotIR, world: WorldModel) -> Tuple[List[str], List[str]]:
    """
    Returns (errors, warnings)
    - errors: must-fix inconsistencies
    - warnings: potential mismatches that are tolerated unless --strict
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Precompute world frame registry paths for flexible matching
    world_frame_registry_paths = {f.registry_path for f in world.frames.values()}

    # 1) Objects: check aliases referenced in IR.world_binding.objects
    for object_alias, object_ref in ir.world_binding.objects.items():
        if world.has_object(object_alias):
            continue
        # Try to match by registry id fallback: IR may store registry id (e.g., "obj_block_red_01")
        # and world.frames.registry_path may contain that registry id (e.g., "obj_block_red_01/frame").
        registry_base = object_ref.split("/")[0]
        matched = False
        for rp in world_frame_registry_paths:
            if rp.startswith(registry_base):
                matched = True
                break
        if matched:
            warnings.append(
                f"IR object alias '{object_alias}' not found in world.objects but registry match for '{object_ref}' found in world.frames"
            )
        else:
            errors.append(f"IR object alias '{object_alias}' not found in world.objects")

    # 2) Frames: allow match by alias or by registry_path
    for frame_alias, frame_ref in ir.world_binding.frames.items():
        if frame_alias == world.world_frame:
            continue
        if world.has_frame(frame_alias):
            continue
        # check if the IR-provided frame_ref (registry path) exists among world's registry paths
        # accept if any world's registry_path equals or startswith the frame_ref (be flexible)
        matched = any(rp == frame_ref or rp.startswith(frame_ref.split("/")[0]) for rp in world_frame_registry_paths)
        if not matched:
            errors.append(f"IR frame alias '{frame_alias}' not found in world.frames (looked for '{frame_ref}')")

    # 3) Features: check existence and parent consistency
    for feature_alias, feature_binding in ir.world_binding.features.items():
        if not world.has_feature(feature_alias):
            # it's possible that world defines a feature under a different id; mark as warning if possible
            warnings.append(f"IR feature alias '{feature_alias}' not found in world.features")
            continue
        world_feature = world.get_feature(feature_alias)
        if world_feature.parent_object != feature_binding.parent_object:
            errors.append(
                f"feature '{feature_alias}' parent mismatch: IR={feature_binding.parent_object}, world={world_feature.parent_object}"
            )
        # If feature specifies a frame, ensure frame exists (either in IR alias or world frames)
        if feature_binding.frame:
            fb_frame = feature_binding.frame
            if fb_frame not in ir.world_binding.frames and not world.has_frame(fb_frame):
                # also permit registry-path match
                matched = any(rp == fb_frame or rp.startswith(fb_frame.split("/")[0]) for rp in world_frame_registry_paths)
                if not matched:
                    warnings.append(f"feature '{feature_alias}' declares frame '{fb_frame}' which is unknown in IR/world")

    return errors, warnings


def format_bool(flag: bool) -> str:
    return "YES" if flag else "NO"


def print_human_summary(sample_path: str, ir: GenericCobotIR, result: VerificationResult, world: WorldModel | None = None) -> None:
    print("[SUMMARY]")
    print(f"sample={sample_path}")
    print(f"task={ir.task_id} | goal={ir.task_spec.goal} | status={result.status.upper()}")
    if world is not None:
        print(
            "world="
            f"objects:{len(world.objects)}"
            f" | features:{len(world.features)}"
            f" | attached_object:{world.robot_state.attached_object or 'none'}"
        )
    print(
        "execution="
        f"{result.summary.steps_executed}/{result.summary.steps_total} steps"
        f" | goal_reached={format_bool(result.summary.goal_reached)}"
        f" | collision={format_bool(result.summary.collision_detected)}"
    )

    if result.metrics is not None:
        min_clearance = (
            f"{result.metrics.min_clearance:.4f} m"
            if result.metrics.min_clearance is not None
            else "n/a"
        )
        max_velocity = (
            f"{result.metrics.max_joint_velocity_ratio:.2f}"
            if result.metrics.max_joint_velocity_ratio is not None
            else "n/a"
        )
        est_time = (
            f"{result.metrics.estimated_execution_time_sec:.1f} s"
            if result.metrics.estimated_execution_time_sec is not None
            else "n/a"
        )
        print(
            "metrics="
            f"min_clearance:{min_clearance}"
            f" | max_joint_velocity_ratio:{max_velocity}"
            f" | est_time:{est_time}"
        )

    if result.errors:
        print("errors:")
        for error in result.errors:
            print(
                f"  - {error.step_id} [{error.type}] {error.message}"
            )

    print("steps:")
    for trace in result.traces:
        status = "OK" if trace.success else "FAIL"
        line = f"  - {trace.step_id} {trace.step_type}: {status}"
        if trace.error:
            line += f" | {trace.error['type']}: {trace.error['message']}"
        print(line)


def discover_samples(samples_dir: str) -> list[str]:
    p = Path(samples_dir)
    if not p.exists() or not p.is_dir():
        print(f"[WARN] samples directory not found: {samples_dir}. Falling back to default sample list.")
        return []
    # choose json files that are not world.json
    return sorted([str(x) for x in p.glob("*.json") if not x.name.endswith(".world.json")])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run demo IR samples with optional world models")
    parser.add_argument("--samples-dir", type=str, default="samples", help="Directory containing sample IR/world JSON files")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors in world consistency checks")
    parser.add_argument("--world", type=str, default=None, help="Optional: force a single world file path for all samples")
    args = parser.parse_args()

    # default set if samples directory missing or empty
    default_sample_files = [
        "samples/sample_pick_place.json",
        "samples/sample_pick_place_target_surface.json",
        "samples/sample_pick_place_target_slot.json",
        "samples/sample_pick_place_fixture_pocket.json",
        "samples/sample_insert.json",
        "samples/sample_insert_failure_ir.json",
    ]

    samples = discover_samples(args.samples_dir)
    if not samples:
        samples = default_sample_files

    for sample_path in samples:
        print("=" * 80)
        print(f"[LOAD] {sample_path}")
        try:
            ir = load_ir(sample_path)
        except Exception:
            print(f"[SKIP] failed to load IR: {sample_path}")
            continue

        print(f"task_id={ir.task_id}")
        print(f"goal={ir.task_spec.goal}")
        print(f"steps={len(ir.action_plan)}")

        # world selection
        world_path = None
        if args.world:
            world_path = Path(args.world)
            if not world_path.exists():
                print(f"[WARN] specified world file {args.world} does not exist -> running IR only")
                world_path = None
        else:
            world_path = resolve_world_path(sample_path)

        world = None
        if world_path is not None:
            print(f"[LOAD WORLD] {world_path}")
            try:
                world = load_world(str(world_path))
            except Exception:
                print(f"[SKIP] failed to load world: {world_path}")
                world = None

            if world is not None:
                errors, warnings = validate_ir_world_consistency(ir, world)
                if errors:
                    print("[WORLD CHECK] inconsistent (errors):")
                    for issue in errors:
                        print(f"  - {issue}")
                    print("[SKIP] skipping execution due to errors in world consistency")
                    continue
                if warnings:
                    if args.strict:
                        print("[WORLD CHECK] warnings treated as errors (--strict):")
                        for issue in warnings:
                            print(f"  - {issue}")
                        print("[SKIP] skipping execution due to strict mode")
                        continue
                    else:
                        print("[WORLD CHECK] warnings:")
                        for issue in warnings:
                            print(f"  - {issue}")
                else:
                    print("[WORLD CHECK] consistent")
        else:
            print("[LOAD WORLD] not found -> run with IR only")

        result = run_ir(ir, world_model=world)

        print_human_summary(sample_path, ir, result, world=world)
        print("[RESULT]")
        print(result.model_dump_json(indent=2, exclude_none=True))


if __name__ == "__main__":
    main()