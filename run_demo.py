from __future__ import annotations

import json
from pathlib import Path

from executor import run_ir
from ir_models import GenericCobotIR, VerificationResult
from world_model import WorldModel


def load_ir(path: str) -> GenericCobotIR:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return GenericCobotIR.model_validate(data)


def load_world(path: str) -> WorldModel:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return WorldModel.model_validate(data)


def resolve_world_path(ir_path: str) -> Path | None:
    ir_file = Path(ir_path)
    if ir_file.suffix != ".json":
        return None
    candidate = ir_file.with_name(f"{ir_file.stem}.world.json")
    return candidate if candidate.exists() else None


def validate_ir_world_consistency(ir: GenericCobotIR, world: WorldModel) -> list[str]:
    issues: list[str] = []

    for object_alias in ir.world_binding.objects.keys():
        if not world.has_object(object_alias):
            issues.append(f"IR object alias '{object_alias}' not found in world.objects")

    for frame_alias in ir.world_binding.frames.keys():
        if frame_alias == world.world_frame:
            continue
        if not world.has_frame(frame_alias):
            issues.append(f"IR frame alias '{frame_alias}' not found in world.frames")

    for feature_alias, feature_binding in ir.world_binding.features.items():
        if not world.has_feature(feature_alias):
            issues.append(f"IR feature alias '{feature_alias}' not found in world.features")
            continue
        world_feature = world.get_feature(feature_alias)
        if world_feature.parent_object != feature_binding.parent_object:
            issues.append(
                f"feature '{feature_alias}' parent mismatch: IR={feature_binding.parent_object}, world={world_feature.parent_object}"
            )

    return issues


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


def main() -> None:
    sample_files = [
        "samples/sample_pick_place.json",
        "samples/sample_pick_place_target_surface.json",
        "samples/sample_pick_place_target_slot.json",
        "samples/sample_pick_place_fixture_pocket.json",
        "samples/sample_insert.json",
        "samples/sample_insert_failure_ir.json",
    ]

    for sample_path in sample_files:
        print("=" * 80)
        print(f"[LOAD] {sample_path}")
        ir = load_ir(sample_path)

        print(f"task_id={ir.task_id}")
        print(f"goal={ir.task_spec.goal}")
        print(f"steps={len(ir.action_plan)}")

        world_path = resolve_world_path(sample_path)
        world = None
        if world_path is not None:
            print(f"[LOAD WORLD] {world_path}")
            world = load_world(str(world_path))
            issues = validate_ir_world_consistency(ir, world)
            if issues:
                print("[WORLD CHECK] inconsistent")
                for issue in issues:
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
