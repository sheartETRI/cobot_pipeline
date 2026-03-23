#!/usr/bin/env python3
"""Generate pick/place IR and optionally run it against a world model."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

from pydantic import ValidationError

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ir_models import GenericCobotIR
from sample_paths import (
    ensure_sample_dirs,
    exported_world_sample_path,
    ir_sample_path,
    world_sample_path,
)
from world_model import WorldModel

try:
    from scripts import pybullet_world_utils
except ImportError:
    import pybullet_world_utils


ensure_sample_dirs()

DEFAULT_BLOCK_POS = (0.3, 0.1, 0.02)
DEFAULT_TARGET_POS = (0.55, 0.12, 0.01)
REQUIRED_OBJECTS = ("block_red", "target_zone")
REQUIRED_FEATURES = ("block_red_top_surface", "target_surface")
REQUIRED_FRAMES = ("block_red_frame", "target_zone_frame", "home_frame")


def print_info(message: str) -> None:
    print(f"[INFO] {message}")


def print_warn(message: str) -> None:
    print(f"[WARN] {message}")


def print_error(message: str) -> None:
    print(f"[ERROR] {message}")


def print_auto_fill(name: str, reason: str, payload: dict[str, Any]) -> None:
    print_info(f"Auto-generated '{name}' from {reason}: {json.dumps(payload, ensure_ascii=False)}")


def format_validation_error(error: ValidationError) -> str:
    parts: list[str] = []
    for issue in error.errors():
        loc = ".".join(str(item) for item in issue.get("loc", []))
        msg = issue.get("msg", "validation error")
        parts.append(f"{loc}: {msg}")
    return "; ".join(parts)


def resolve_prefix(prefix: str, force: bool) -> str:
    ir_path = ir_sample_path(prefix)
    world_path = world_sample_path(prefix)
    if force:
        return prefix
    if not ir_path.exists() and not world_path.exists():
        return prefix
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    new_prefix = f"{prefix}-{ts}"
    print_info(f"'{prefix}' already exists, using '{new_prefix}' instead")
    return new_prefix


def parse_positions(nl: str) -> list[Tuple[float, float, float]]:
    coords = re.findall(
        r"\(?\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)?",
        nl,
    )
    return [(float(x), float(y), float(z)) for x, y, z in coords]


def default_world_dict(
    block_pos: Tuple[float, float, float],
    target_pos: Tuple[float, float, float],
) -> dict[str, Any]:
    return {
        "world_frame": "world",
        "objects": {
            "block_red": {
                "object_id": "block_red",
                "object_type": "block",
                "pose": {
                    "frame": "world",
                    "position": list(block_pos),
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                },
                "geometry": {"type": "box", "size": [0.04, 0.04, 0.04]},
                "movable": True,
                "graspable": True,
                "collision_enabled": True,
                "metadata": {"color": "red", "registry_id": "obj_block_red_01"},
            },
            "target_zone": {
                "object_id": "target_zone",
                "object_type": "fixture",
                "pose": {
                    "frame": "world",
                    "position": list(target_pos),
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                },
                "geometry": {"type": "box", "size": [0.2, 0.15, 0.02]},
                "movable": False,
                "graspable": False,
                "collision_enabled": True,
                "metadata": {"role": "placement_fixture", "registry_id": "obj_target_zone_01"},
            },
        },
        "features": {
            "block_red_top_surface": {
                "feature_id": "block_red_top_surface",
                "parent_object": "block_red",
                "feature_type": "surface",
                "local_pose": {
                    "frame": "block_red",
                    "position": [0.0, 0.0, 0.02],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                },
                "size_hint": [0.04, 0.04],
                "metadata": {"usage": "grasp_reference_surface"},
            },
            "target_surface": {
                "feature_id": "target_surface",
                "parent_object": "target_zone",
                "feature_type": "surface",
                "local_pose": {
                    "frame": "target_zone",
                    "position": [0.0, 0.0, 0.01],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                },
                "size_hint": [0.18, 0.12],
                "metadata": {"usage": "flat_placement_surface"},
            },
        },
        "frames": {
            "block_red_frame": {
                "frame_id": "block_red_frame",
                "registry_path": "obj_block_red_01/frame",
                "pose": {
                    "frame": "world",
                    "position": list(block_pos),
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                },
                "metadata": {"source": "object_pose"},
            },
            "target_zone_frame": {
                "frame_id": "target_zone_frame",
                "registry_path": "obj_target_zone_01/frame",
                "pose": {
                    "frame": "world",
                    "position": list(target_pos),
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                },
                "metadata": {"source": "object_pose"},
            },
            "home_frame": {
                "frame_id": "home_frame",
                "registry_path": "robot/home_pose",
                "pose": {
                    "frame": "world",
                    "position": [0.4, 0.0, 0.3],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                },
                "metadata": {"source": "robot_home"},
            },
        },
        "robot_state": {
            "base_frame": "world",
            "tcp_pose": {
                "frame": "world",
                "position": [0.4, 0.0, 0.3],
                "orientation": [0.0, 0.0, 0.0, 1.0],
            },
            "joint_positions": [0.0, -1.2, 1.5, 0.0, 1.2, 0.0],
            "attached_object": None,
        },
    }


def validate_world_dict(world_data: dict[str, Any]) -> WorldModel:
    try:
        return WorldModel.model_validate(world_data)
    except ValidationError as error:
        raise ValueError(f"WorldModel validation failed: {format_validation_error(error)}") from error


def validate_ir_dict(ir_data: dict[str, Any]) -> GenericCobotIR:
    try:
        return GenericCobotIR.model_validate(ir_data)
    except ValidationError as error:
        raise ValueError(f"GenericCobotIR validation failed: {format_validation_error(error)}") from error


def load_world_file(path: str) -> WorldModel:
    world_path = Path(path)
    if not world_path.exists():
        raise FileNotFoundError(f"world file not found: {world_path}")
    try:
        world_data = json.loads(world_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in world file '{world_path}': {error}") from error
    world = validate_world_dict(world_data)
    print_info(
        f"Loaded world '{world_path}' "
        f"(objects={len(world.objects)}, features={len(world.features)}, frames={len(world.frames)})"
    )
    return world


def geometry_size(geometry: Any) -> list[float]:
    geometry_type = getattr(geometry, "type", None)
    if geometry_type == "box":
        return list(geometry.size)
    if geometry_type == "cylinder":
        return [geometry.radius * 2.0, geometry.radius * 2.0, geometry.height]
    if geometry_type == "plane":
        return [geometry.size[0], geometry.size[1], 0.002]
    return [0.04, 0.04, 0.04]


def ensure_object_aliases(world: WorldModel) -> None:
    for object_id in REQUIRED_OBJECTS:
        if object_id not in world.objects:
            raise ValueError(
                f"existing world does not contain required object alias '{object_id}'. "
                "Automatic alias guessing is disabled."
            )


def ensure_world_features_and_frames(
    world: WorldModel,
    desired_target: Tuple[float, float, float] | None = None,
) -> WorldModel:
    ensure_object_aliases(world)
    world_data = world.model_dump(mode="python")

    block_obj = world.get_object("block_red")
    target_obj = world.get_object("target_zone")
    block_size = geometry_size(block_obj.geometry)
    target_size = geometry_size(target_obj.geometry)
    block_registry_id = block_obj.metadata.get("registry_id")
    target_registry_id = target_obj.metadata.get("registry_id")

    if not block_registry_id or not target_registry_id:
        raise ValueError(
            "existing world must provide metadata.registry_id for 'block_red' and 'target_zone'"
        )

    frames = world_data.setdefault("frames", {})
    if "block_red_frame" not in frames:
        frame_payload = {
            "frame_id": "block_red_frame",
            "registry_path": f"{block_registry_id}/frame",
            "pose": world_data["objects"]["block_red"]["pose"],
            "metadata": {"source": "auto_generated_from_object"},
        }
        print_warn("existing world is missing 'block_red_frame'; generating it from object pose")
        print_auto_fill("block_red_frame", "object pose", frame_payload)
        frames["block_red_frame"] = frame_payload
    if "target_zone_frame" not in frames:
        frame_payload = {
            "frame_id": "target_zone_frame",
            "registry_path": f"{target_registry_id}/frame",
            "pose": world_data["objects"]["target_zone"]["pose"],
            "metadata": {"source": "auto_generated_from_object"},
        }
        print_warn("existing world is missing 'target_zone_frame'; generating it from object pose")
        print_auto_fill("target_zone_frame", "object pose", frame_payload)
        frames["target_zone_frame"] = frame_payload
    if "home_frame" not in frames:
        frame_payload = {
            "frame_id": "home_frame",
            "registry_path": "robot/home_pose",
            "pose": world_data["robot_state"]["tcp_pose"],
            "metadata": {"source": "auto_generated_from_robot_state"},
        }
        print_warn("existing world is missing 'home_frame'; generating it from robot_state.tcp_pose")
        print_auto_fill("home_frame", "robot_state.tcp_pose", frame_payload)
        frames["home_frame"] = frame_payload

    features = world_data.setdefault("features", {})
    if "block_red_top_surface" not in features:
        feature_payload = {
            "feature_id": "block_red_top_surface",
            "parent_object": "block_red",
            "feature_type": "surface",
            "local_pose": {
                "frame": "block_red",
                "position": [0.0, 0.0, block_size[2] / 2.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
            },
            "size_hint": [block_size[0], block_size[1]],
            "metadata": {"source": "auto_generated"},
        }
        print_warn("existing world is missing 'block_red_top_surface'; generating a top surface feature")
        print_auto_fill("block_red_top_surface", "block_red geometry top face", feature_payload)
        features["block_red_top_surface"] = feature_payload

    if "target_surface" not in features:
        feature_payload = {
            "feature_id": "target_surface",
            "parent_object": "target_zone",
            "feature_type": "surface",
            "local_pose": {
                "frame": "target_zone",
                "position": [0.0, 0.0, target_size[2] / 2.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
            },
            "size_hint": [target_size[0], target_size[1]],
            "width": target_size[0],
            "depth": target_size[2],
            "metadata": {"source": "auto_generated"},
        }
        print_warn("existing world is missing 'target_surface'; generating a top surface feature")
        print_auto_fill("target_surface", "target_zone geometry top face", feature_payload)
        features["target_surface"] = feature_payload

    if desired_target is not None:
        target_pose = world.get_object("target_zone").pose
        local_position = [
            desired_target[0] - target_pose.position[0],
            desired_target[1] - target_pose.position[1],
            desired_target[2] - target_pose.position[2],
        ]
        features["target_surface"]["local_pose"] = {
            "frame": "target_zone",
            "position": local_position,
            "orientation": [0.0, 0.0, 0.0, 1.0],
        }
        print_info(
            "Adjusted target_surface local pose from natural-language target "
            f"to local offset {local_position}"
        )

    return validate_world_dict(world_data)


def build_ir_from_world(
    nl: str,
    world: WorldModel,
    task_suffix: str = "",
) -> dict[str, Any]:
    block_registry_id = world.get_object("block_red").metadata.get("registry_id")
    target_registry_id = world.get_object("target_zone").metadata.get("registry_id")
    if not block_registry_id or not target_registry_id:
        raise ValueError(
            "existing world must provide metadata.registry_id for 'block_red' and 'target_zone'"
        )

    now = datetime.now(timezone.utc).isoformat()
    task_id = (
        f"task_pick_place_{uuid.uuid4().hex[:8]}"
        if not task_suffix
        else f"task_{task_suffix}_{uuid.uuid4().hex[:6]}"
    )
    base_frame = world.robot_state.base_frame
    tool_frame = world.robot_state.tcp_pose.frame or base_frame

    return {
        "ir_version": "0.1",
        "task_id": task_id,
        "created_by": "nl_generator",
        "created_at": now,
        "task_spec": {
            "goal": "pick_and_place",
            "command_text": nl,
            "priority": "normal",
            "success_condition": ["object_at_destination:block_red:target_surface"],
            "assumptions": [],
        },
        "robot_profile": {
            "robot_type": "generic_cobot",
            "arm_dof": max(1, len(world.robot_state.joint_positions) or 6),
            "has_gripper": True,
            "tool_frame": tool_frame,
            "base_frame": base_frame,
            "motion_limits_profile": "default_cobot",
        },
        "world_binding": {
            "scene_id": "existing_world_scene",
            "objects": {
                "block_red": block_registry_id,
                "target_zone": target_registry_id,
            },
            "frames": {
                "block_red_frame": world.get_frame("block_red_frame").registry_path,
                "target_zone_frame": world.get_frame("target_zone_frame").registry_path,
                "home_frame": world.get_frame("home_frame").registry_path,
            },
            "regions": {},
            "features": {
                "block_red_top_surface": {
                    "parent_object": "block_red",
                    "feature_type": "surface",
                    "frame": "block_red_frame",
                    "description": "top surface",
                },
                "target_surface": {
                    "parent_object": "target_zone",
                    "feature_type": "surface",
                    "frame": "target_zone_frame",
                    "description": "placement surface",
                },
            },
        },
        "action_plan": [
            {"step_id": "s1", "type": "find_object", "inputs": {"object": "block_red"}},
            {
                "step_id": "s2",
                "type": "approach",
                "inputs": {
                    "target_object": "block_red",
                    "target_feature": "block_red_top_surface",
                    "approach_pose": {
                        "ref": "block_red_frame",
                        "offset": [0.0, 0.0, 0.08],
                        "orientation_policy": "align_with_object_top",
                    },
                },
            },
            {
                "step_id": "s3",
                "type": "grasp",
                "inputs": {
                    "target_object": "block_red",
                    "target_feature": "block_red_top_surface",
                    "grasp_mode": "pinch",
                },
            },
            {"step_id": "s4", "type": "retreat", "inputs": {"direction": "tool_z_negative", "distance": 0.08}},
            {
                "step_id": "s5",
                "type": "move_linear",
                "inputs": {
                    "target_pose": {
                        "ref": "target_zone_frame",
                        "offset": [0.0, 0.0, 0.1],
                        "orientation_policy": "keep_current",
                    }
                },
            },
            {
                "step_id": "s6",
                "type": "place",
                "inputs": {
                    "target_object": "block_red",
                    "target_feature": "target_surface",
                    "destination_pose": {
                        "ref": "target_zone_frame",
                        "offset": [0.0, 0.0, 0.0],
                        "orientation_policy": "align_to_target_surface",
                    },
                },
            },
            {"step_id": "s7", "type": "release", "inputs": {"target_object": "block_red"}},
            {"step_id": "s8", "type": "retreat", "inputs": {"direction": "tool_z_negative", "distance": 0.1}},
        ],
        "verification_policy": {
            "collision_check": True,
            "ik_check": True,
            "joint_limit_check": True,
            "velocity_limit_check": True,
            "force_limit_check": False,
            "max_retry": 3,
            "acceptance_rules": [],
        },
        "repair_state": {"retry_count": 0, "last_error": None, "repair_history": []},
    }


def save_world_model(world: WorldModel, path: Path) -> str:
    path.write_text(json.dumps(world.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
    print_info(f"Saved WorldModel JSON: {path}")
    return str(path)


def save_ir_model(ir: GenericCobotIR, path: Path) -> str:
    path.write_text(json.dumps(ir.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
    print_info(f"Saved GenericCobotIR JSON: {path}")
    return str(path)


def print_world_binding_summary(ir: GenericCobotIR) -> None:
    print_info(
        "IR world_binding summary: "
        f"objects={dict(ir.world_binding.objects)} "
        f"frames={dict(ir.world_binding.frames)} "
        f"features={list(ir.world_binding.features.keys())}"
    )


def run_sim(
    ir_path: str,
    world_path: str | None,
    sim_backend: str = "pybullet",
    gui: bool = False,
    step_wait: bool = False,
) -> None:
    cmd = [
        sys.executable,
        str(ROOT_DIR / "run_demo.py"),
        "--sample",
        ir_path,
        "--sim-backend",
        sim_backend,
    ]
    if world_path is not None:
        cmd.extend(["--world", world_path])
    if gui:
        cmd.append("--pybullet-gui")
    if step_wait:
        cmd.append("--pybullet-step-wait")
    print_info(f"Running simulation command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT_DIR)


def export_world_from_pybullet(prefix: str) -> WorldModel:
    export_path = exported_world_sample_path(prefix)
    print_info(f"Exporting current PyBullet session to {export_path}")
    exported = pybullet_world_utils.export_world_model(str(export_path))
    return load_world_file(exported)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pick/place IR and optionally run it")
    parser.add_argument(
        "--nl",
        type=str,
        default=None,
        help="Natural-language command, e.g. 'pick red block at (0.3,0.1,0.02) and place it at (0.55,0.12,0.01)'",
    )
    parser.add_argument("--out-prefix", type=str, default="sample_pick_place", help="Output file prefix")
    parser.add_argument("--no-run", action="store_true", help="Generate JSON files only")
    parser.add_argument("--sim-backend", type=str, default="pybullet", choices=["pybullet", "mock"], help="Simulator backend")
    parser.add_argument("--pybullet-gui", action="store_true", help="Enable PyBullet GUI")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output filenames")
    parser.add_argument("--step-by-step", action="store_true", help="Wait for Enter between PyBullet steps")
    parser.add_argument("--existing-world", type=str, default=None, help="Use an existing WorldModel JSON file")
    parser.add_argument(
        "--world-from-pybullet",
        action="store_true",
        help="Export the current PyBullet session to WorldModel JSON and use it as the world",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.existing_world and args.world_from_pybullet:
        print_error("Use either --existing-world or --world-from-pybullet, not both")
        return 2
    if args.step_by_step and not args.pybullet_gui:
        print_warn("--step-by-step has effect only with --pybullet-gui")

    if args.nl:
        nl_text = args.nl
    else:
        nl_text = input(
            "Natural-language command (example: pick red block at (0.3,0.1,0.02) and place it at (0.55,0.12,0.01)):\n"
        )

    coords = parse_positions(nl_text)
    desired_block = coords[0] if len(coords) >= 1 else DEFAULT_BLOCK_POS
    desired_target = coords[1] if len(coords) >= 2 else DEFAULT_TARGET_POS
    if len(coords) < 2:
        print_warn(
            "Could not extract two coordinate triples from input; using default target "
            f"{DEFAULT_TARGET_POS}"
        )

    prefix = resolve_prefix(args.out_prefix, force=args.force)

    try:
        if args.world_from_pybullet:
            base_world = export_world_from_pybullet(prefix)
        elif args.existing_world:
            base_world = load_world_file(args.existing_world)
        else:
            print_info("No existing world provided; generating a new default world")
            base_world = validate_world_dict(default_world_dict(desired_block, desired_target))

        using_existing_world = args.world_from_pybullet or args.existing_world is not None
        if using_existing_world:
            print_info("Using existing world data; skipping default world generation")
            world_model = ensure_world_features_and_frames(base_world, desired_target=desired_target)
        else:
            world_model = base_world

        ir_data = build_ir_from_world(nl_text, world_model)
        ir_model = validate_ir_dict(ir_data)

        world_path = save_world_model(world_model, world_sample_path(prefix))
        ir_path = save_ir_model(ir_model, ir_sample_path(prefix))

        print_info(
            f"Validation complete: objects={len(world_model.objects)}, "
            f"features={len(world_model.features)}, frames={len(world_model.frames)}"
        )
        print_world_binding_summary(ir_model)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print_error(str(error))
        return 1
    except Exception as error:
        print_error(f"Unexpected failure: {error}")
        raise

    if not args.no_run:
        try:
            run_sim(
                ir_path,
                world_path=world_path,
                sim_backend=args.sim_backend,
                gui=args.pybullet_gui,
                step_wait=args.step_by_step,
            )
        except subprocess.CalledProcessError as error:
            print_error(f"run_demo.py exited with status {error.returncode}")
            return error.returncode or 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
