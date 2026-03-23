#!/usr/bin/env python3
"""Run an interactive NL -> IR -> execute session against a persistent world state."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from executor import close_persistent_simulator, run_ir_with_final_world
from sample_paths import ir_sample_path, world_sample_path
from scripts import generate_and_run_rule as generator
from world_model import WorldModel


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive cobot session with persistent world state")
    parser.add_argument("--existing-world", type=str, default=None, help="Initial WorldModel JSON path")
    parser.add_argument("--out-prefix", type=str, default="interactive_session", help="Prefix used when saving generated IR files")
    parser.add_argument("--save-session", action="store_true", help="Keep generated session IR/world artifacts on exit")
    parser.add_argument("--sim-backend", type=str, default="mock", choices=["mock", "pybullet"], help="Simulator backend")
    parser.add_argument("--pybullet-gui", action="store_true", help="Open PyBullet GUI during execution")
    parser.add_argument("--step-by-step", action="store_true", help="Wait for Enter between PyBullet steps")
    parser.add_argument("--pybullet-motion-delay", type=float, default=0.0, help="Sleep duration in seconds after each PyBullet motion sample")
    parser.add_argument("--pybullet-step-pause", type=float, default=0.0, help="Sleep duration in seconds after each PyBullet primitive step")
    parser.add_argument("--save-final-world", type=str, default=None, help="Optional path to save the final world state on exit")
    return parser.parse_args(argv)


def load_initial_world(path: str | None) -> WorldModel:
    if path is not None:
        return generator.load_world_file(path)
    generator.print_info("No initial world provided; creating a default world")
    return generator.validate_world_dict(generator.default_world_dict(generator.DEFAULT_BLOCK_POS, generator.DEFAULT_TARGET_POS))


def split_commands(nl_text: str) -> list[str]:
    return [part.strip() for part in nl_text.split(";") if part.strip()]


def execute_single_command(
    *,
    nl_text: str,
    current_world: WorldModel,
    args: argparse.Namespace,
    command_index: int,
) -> tuple[WorldModel, bool, Path | None]:
    prefix = f"{args.out_prefix}-{command_index:03d}"
    coords = generator.parse_positions(nl_text)
    desired_target = coords[1] if len(coords) >= 2 else generator.DEFAULT_TARGET_POS
    if len(coords) < 2:
        generator.print_warn(
            "Could not extract two coordinate triples from input; using default target "
            f"{generator.DEFAULT_TARGET_POS}"
        )

    try:
        source_object_id = generator.resolve_source_object_alias(current_world, nl_text)
        target_object_id, _ = generator.resolve_target_feature_binding(current_world, nl_text, source_object_id)
        prepared_world = generator.ensure_world_features_and_frames(
            current_world,
            source_object_id=source_object_id,
            target_object_id=target_object_id,
            desired_target=desired_target,
        )
        ir_model = generator.validate_ir_dict(
            generator.build_ir_from_world(nl_text, prepared_world, task_suffix=f"session_{command_index:03d}")
        )
        ir_path = Path(generator.save_ir_model(ir_model, ir_sample_path(prefix)))

        result, final_world = run_ir_with_final_world(
            ir_model,
            world_model=prepared_world,
            simulator_backend=args.sim_backend,
            pybullet_gui=args.pybullet_gui,
            keep_simulator_open=False,
            pybullet_motion_delay=args.pybullet_motion_delay,
            pybullet_step_pause=args.pybullet_step_pause,
            pybullet_step_wait_for_input=args.step_by_step,
        )
        print(f"[RESULT] status={result.status} steps={result.summary.steps_executed}/{result.summary.steps_total}")
        if result.errors:
            for error in result.errors:
                print(f"[ERROR] {error.step_id} {error.type}: {error.message}")
        print(f"[IR] {ir_path}")

        if final_world is not None:
            current_world = final_world
        return current_world, result.status == "passed", ir_path
    except Exception as error:
        generator.print_error(str(error))
        return current_world, False, None
    finally:
        close_persistent_simulator()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    current_world = load_initial_world(args.existing_world)
    command_index = 1
    saved_world_path: Path | None = None
    generated_ir_paths: list[Path] = []

    print("Interactive session started. Type a command, or 'exit' to quit.")
    while True:
        try:
            nl_text = input("nl> ").strip()
        except EOFError:
            print()
            break

        if not nl_text:
            continue
        if nl_text.lower() in {"exit", "quit"}:
            break

        commands = split_commands(nl_text)
        for command in commands:
            current_world, ok, ir_path = execute_single_command(
                nl_text=command,
                current_world=current_world,
                args=args,
                command_index=command_index,
            )
            if ir_path is not None:
                generated_ir_paths.append(ir_path)
            command_index += 1
            if not ok:
                break

    if args.save_final_world:
        saved_world_path = Path(args.save_final_world)
        generator.save_world_model(current_world, saved_world_path)
    elif args.save_session:
        latest_path = world_sample_path(f"{args.out_prefix}_latest")
        generator.save_world_model(current_world, latest_path)
        print(f"[WORLD] latest session world saved to {latest_path}")
    else:
        latest_path = world_sample_path(f"{args.out_prefix}_latest")
        generator.save_world_model(current_world, latest_path)
        saved_world_path = latest_path
        print(f"[WORLD] latest session world saved to {latest_path}")
        if saved_world_path.exists():
            saved_world_path.unlink()
            print(f"[WORLD] deleted transient session world {saved_world_path}")
        for ir_path in generated_ir_paths:
            if ir_path.exists():
                ir_path.unlink()
                print(f"[IR] deleted transient session IR {ir_path}")

    close_persistent_simulator()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
