from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

from pydantic import ValidationError

from executor import close_persistent_simulator, get_persistent_simulator, run_ir
from ir_models import GenericCobotIR, VerificationResult
from world_model import WorldModel


logger = logging.getLogger(__name__)


def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(levelname)s:%(name)s:%(message)s",
        stream=sys.stderr,
        force=True,
    )


def load_ir(path: str) -> GenericCobotIR:
    try:
        logger.debug("Loading IR file: %s", path)
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
        logger.debug("Loading world file: %s", path)
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
    resolved = candidate if candidate.exists() else None
    logger.debug("Resolved world path for %s -> %s", ir_path, resolved)
    return resolved


def validate_ir_world_consistency(ir: GenericCobotIR, world: WorldModel) -> Tuple[List[str], List[str]]:
    """
    Returns (errors, warnings)
    - errors: must-fix inconsistencies
    - warnings: potential mismatches that are tolerated unless --strict
    """
    errors: List[str] = []
    warnings: List[str] = []

    world_frame_registry_paths = {f.registry_path for f in world.frames.values()}

    for object_alias, registry_id in ir.world_binding.objects.items():
        if not world.has_object(object_alias):
            logger.debug("Object alias missing in world.objects: %s", object_alias)
            errors.append(f"IR object alias '{object_alias}' not found in world.objects")
            continue

        world_object = world.get_object(object_alias)
        world_registry_id = world_object.metadata.get("registry_id")
        if world_registry_id is None:
            logger.debug("Object %s missing registry_id in world metadata", object_alias)
            warnings.append(
                f"world object '{object_alias}' is missing metadata.registry_id; expected '{registry_id}'"
            )
            continue
        if world_registry_id != registry_id:
            logger.debug(
                "Object registry mismatch for %s: ir=%s world=%s",
                object_alias,
                registry_id,
                world_registry_id,
            )
            errors.append(
                f"object '{object_alias}' registry_id mismatch: IR={registry_id}, world={world_registry_id}"
            )

    for frame_alias, frame_ref in ir.world_binding.frames.items():
        if frame_alias == world.world_frame:
            continue
        if world.has_frame(frame_alias):
            continue
        matched = any(
            rp == frame_ref or rp.startswith(frame_ref.split("/")[0])
            for rp in world_frame_registry_paths
        )
        if not matched:
            logger.debug("Frame alias missing in world.frames: alias=%s ref=%s", frame_alias, frame_ref)
            errors.append(f"IR frame alias '{frame_alias}' not found in world.frames (looked for '{frame_ref}')")

    for feature_alias, feature_binding in ir.world_binding.features.items():
        if not world.has_feature(feature_alias):
            logger.debug("Feature alias missing in world.features: %s", feature_alias)
            warnings.append(f"IR feature alias '{feature_alias}' not found in world.features")
            continue
        world_feature = world.get_feature(feature_alias)
        if world_feature.parent_object != feature_binding.parent_object:
            logger.debug(
                "Feature parent mismatch for %s: ir=%s world=%s",
                feature_alias,
                feature_binding.parent_object,
                world_feature.parent_object,
            )
            errors.append(
                f"feature '{feature_alias}' parent mismatch: IR={feature_binding.parent_object}, world={world_feature.parent_object}"
            )
        if feature_binding.frame:
            fb_frame = feature_binding.frame
            if fb_frame not in ir.world_binding.frames and not world.has_frame(fb_frame):
                matched = any(rp == fb_frame for rp in world_frame_registry_paths)
                if not matched:
                    logger.debug("Feature frame unresolved: feature=%s frame=%s", feature_alias, fb_frame)
                    warnings.append(f"feature '{feature_alias}' declares frame '{fb_frame}' which is unknown in IR/world")

    logger.debug("World consistency result: errors=%d warnings=%d", len(errors), len(warnings))
    return errors, warnings


def format_bool(flag: bool) -> str:
    return "YES" if flag else "NO"


def print_human_summary(
    sample_path: str,
    ir: GenericCobotIR,
    result: VerificationResult,
    world: WorldModel | None = None,
) -> None:
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
            print(f"  - {error.step_id} [{error.type}] {error.message}")

    print("steps:")
    for trace in result.traces:
        status = "OK" if trace.success else "FAIL"
        line = f"  - {trace.step_id} {trace.step_type}: {status}"
        if trace.error:
            line += f" | {trace.error['type']}: {trace.error['message']}"
        print(line)


def discover_samples(samples_dir: str) -> list[str]:
    sample_dir = Path(samples_dir)
    if not sample_dir.exists() or not sample_dir.is_dir():
        print(f"[WARN] samples directory not found: {samples_dir}. Falling back to default sample list.")
        return []
    samples = sorted(
        str(path)
        for path in sample_dir.glob("*.json")
        if not path.name.endswith(".world.json")
        and not path.name.endswith("_repair_patch.json")
        and not path.name.endswith("_verification.json")
    )
    logger.debug("Discovered samples in %s: %s", samples_dir, samples)
    return samples


def resolve_simulator_backend(requested_backend: str, world: WorldModel | None) -> str:
    if requested_backend == "pybullet" and world is None:
        return "mock"
    return requested_backend


def wait_for_pybullet_gui_close() -> None:
    simulator = get_persistent_simulator()
    if simulator is None:
        return

    print("[PYBULLET GUI] close the window to finish this sample, or press Ctrl+C to stop it here.")
    try:
        while simulator.client_id is not None:
            info = simulator.p.getConnectionInfo(simulator.client_id)
            if not info.get("isConnected", 0):
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[PYBULLET GUI] interrupt received, closing simulator.")
    finally:
        close_persistent_simulator()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run demo IR samples with optional world models")
    parser.add_argument("--samples-dir", type=str, default="samples", help="Directory containing sample IR/world JSON files")
    parser.add_argument("--sample", type=str, default=None, help="Run a single IR sample JSON file")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors in world consistency checks")
    parser.add_argument("--world", type=str, default=None, help="Optional: force a single world file path for all samples")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--sim-backend",
        type=str,
        default="mock",
        choices=["mock", "pybullet"],
        help="Simulator backend",
    )
    parser.add_argument(
        "--pybullet-gui",
        action="store_true",
        help="Open PyBullet GUI when --sim-backend pybullet is selected",
    )
    parser.add_argument(
        "--pybullet-motion-delay",
        type=float,
        default=0.0,
        help="Sleep duration in seconds after each PyBullet motion sample in GUI mode",
    )
    parser.add_argument(
        "--pybullet-step-pause",
        type=float,
        default=0.0,
        help="Sleep duration in seconds after each PyBullet primitive step in GUI mode",
    )
    parser.add_argument(
        "--pybullet-step-wait",
        action="store_true",
        help="Wait for Enter between PyBullet primitive steps in GUI mode",
    )
    args = parser.parse_args()

    configure_logging(args.log_level)
    logger.info(
        "Starting run_demo with samples_dir=%s sample=%s strict=%s world=%s log_level=%s",
        args.samples_dir,
        args.sample,
        args.strict,
        args.world,
        args.log_level,
    )

    default_sample_files = [
        "samples/sample_pick_place.json",
        "samples/sample_pick_place_target_surface.json",
        "samples/sample_pick_place_target_slot.json",
        "samples/sample_pick_place_fixture_pocket.json",
        "samples/sample_insert.json",
        "samples/sample_insert_failure_ir.json",
    ]

    if args.sample:
        samples = [args.sample]
    else:
        samples = discover_samples(args.samples_dir)
        if not samples:
            samples = default_sample_files

    for sample_path in samples:
        logger.info("Processing sample: %s", sample_path)
        print("=" * 80)
        print(f"[LOAD] {sample_path}")
        try:
            ir = load_ir(sample_path)
        except Exception:
            logger.exception("Failed to load IR sample: %s", sample_path)
            print(f"[SKIP] failed to load IR: {sample_path}")
            continue

        print(f"task_id={ir.task_id}")
        print(f"goal={ir.task_spec.goal}")
        print(f"steps={len(ir.action_plan)}")

        world_path = None
        if args.world:
            world_path = Path(args.world)
            if not world_path.exists():
                logger.warning("Specified world file does not exist: %s", args.world)
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
                logger.exception("Failed to load world file: %s", world_path)
                print(f"[SKIP] failed to load world: {world_path}")
                world = None

            if world is not None:
                errors, warnings = validate_ir_world_consistency(ir, world)
                if errors:
                    logger.error("World consistency errors for %s: %s", sample_path, errors)
                    print("[WORLD CHECK] inconsistent (errors):")
                    for issue in errors:
                        print(f"  - {issue}")
                    print("[SKIP] skipping execution due to errors in world consistency")
                    continue
                if warnings:
                    logger.warning("World consistency warnings for %s: %s", sample_path, warnings)
                    if args.strict:
                        print("[WORLD CHECK] warnings treated as errors (--strict):")
                        for issue in warnings:
                            print(f"  - {issue}")
                        print("[SKIP] skipping execution due to strict mode")
                        continue
                    print("[WORLD CHECK] warnings:")
                    for issue in warnings:
                        print(f"  - {issue}")
                else:
                    logger.info("World consistency passed for %s", sample_path)
                    print("[WORLD CHECK] consistent")
        else:
            logger.info("No world file found for %s; running IR only", sample_path)
            print("[LOAD WORLD] not found -> run with IR only")

        simulator_backend = resolve_simulator_backend(args.sim_backend, world)
        if simulator_backend != args.sim_backend:
            logger.info(
                "Falling back simulator backend for %s: requested=%s resolved=%s",
                sample_path,
                args.sim_backend,
                simulator_backend,
            )
            print(
                f"[SIM BACKEND] requested={args.sim_backend} -> using={simulator_backend} "
                "(world model required for pybullet)"
            )

        logger.debug(
            "Running IR task_id=%s with world=%s simulator_backend=%s",
            ir.task_id,
            world_path,
            simulator_backend,
        )
        try:
            result = run_ir(
                ir,
                world_model=world,
                simulator_backend=simulator_backend,
                pybullet_gui=args.pybullet_gui,
                keep_simulator_open=args.pybullet_gui,
                pybullet_motion_delay=args.pybullet_motion_delay,
                pybullet_step_pause=args.pybullet_step_pause,
                pybullet_step_wait_for_input=args.pybullet_step_wait,
            )
        except Exception:
            logger.exception("Simulator backend failed for sample: %s", sample_path)
            print(f"[SKIP] simulator backend failed: {simulator_backend}")
            continue
        logger.info(
            "Completed sample %s with status=%s goal_reached=%s collision=%s",
            sample_path,
            result.status,
            result.summary.goal_reached,
            result.summary.collision_detected,
        )

        print_human_summary(sample_path, ir, result, world=world)
        print("[RESULT]")
        print(result.model_dump_json(indent=2, exclude_none=True))

        if simulator_backend == "pybullet" and args.pybullet_gui:
            wait_for_pybullet_gui_close()


if __name__ == "__main__":
    main()
