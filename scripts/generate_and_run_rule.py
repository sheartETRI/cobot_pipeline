#!/usr/bin/env python3
"""Generate pick/place IR and optionally run it against a world model."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
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
from geometry_utils import compose_pose
from world_model import WorldModel

try:
    from scripts import pybullet_world_utils
except ImportError:
    import pybullet_world_utils


ensure_sample_dirs()

DEFAULT_BLOCK_POS = (0.3, 0.1, 0.02)
DEFAULT_TARGET_POS = (0.55, 0.12, 0.01)
TARGET_OBJECT_ID = "target_zone"
START_TCP_FRAME_ID = "start_tcp_frame"
PREGRASP_CLEARANCE = 0.02
PREPLACE_CLEARANCE = 0.04
GRIPPER_HALF_HEIGHT = 0.06
SUPPORT_CLEARANCE = 0.005


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


def resolve_pose_to_world(world: WorldModel, pose: Any, visited: set[str] | None = None) -> dict[str, Any]:
    if hasattr(pose, "model_dump"):
        pose_data = pose.model_dump(mode="python")
    else:
        pose_data = dict(pose)

    frame_name = pose_data.get("frame", world.world_frame)
    if frame_name == world.world_frame:
        return {
            "frame": world.world_frame,
            "position": list(pose_data["position"]),
            "orientation": list(pose_data["orientation"]),
        }

    next_visited = set() if visited is None else set(visited)
    if frame_name in next_visited:
        raise ValueError(f"cyclic frame reference detected while resolving '{frame_name}'")
    next_visited.add(frame_name)

    if world.has_object(frame_name):
        parent_pose = world.get_object(frame_name).pose
    elif world.has_feature(frame_name):
        parent_pose = world.get_feature(frame_name).local_pose
    elif world.has_frame(frame_name):
        parent_pose = world.get_frame(frame_name).pose
    else:
        raise ValueError(f"unknown pose frame '{frame_name}' in world model")

    parent_world = resolve_pose_to_world(world, parent_pose, visited=next_visited)
    return compose_pose(parent_world, pose_data)


def resolve_feature_pose_to_world(world: WorldModel, feature_id: str) -> dict[str, Any]:
    return resolve_pose_to_world(world, world.get_feature(feature_id).local_pose)


def infer_support_top_z(world: WorldModel, object_id: str) -> float | None:
    source_pose = resolve_pose_to_world(world, world.get_object(object_id).pose)
    source_size = geometry_size(world.get_object(object_id).geometry)
    source_bottom_z = source_pose["position"][2] - (source_size[2] / 2.0)
    source_x, source_y = source_pose["position"][0], source_pose["position"][1]

    best_top_z: float | None = None
    best_gap: float | None = None
    for candidate_id, candidate in world.objects.items():
        if candidate_id == object_id:
            continue
        candidate_pose = resolve_pose_to_world(world, candidate.pose)
        candidate_size = geometry_size(candidate.geometry)
        candidate_top_z = candidate_pose["position"][2] + (candidate_size[2] / 2.0)
        if candidate_top_z > source_bottom_z + 1e-6:
            continue

        half_x = candidate_size[0] / 2.0
        half_y = candidate_size[1] / 2.0
        if abs(source_x - candidate_pose["position"][0]) > half_x:
            continue
        if abs(source_y - candidate_pose["position"][1]) > half_y:
            continue

        gap = source_bottom_z - candidate_top_z
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_top_z = candidate_top_z
    return best_top_z


def resolve_source_object_alias(world: WorldModel, nl: str) -> str:
    normalized = nl.lower().replace("_", " ")
    explicit_pick_match = re.search(r"pick\s+(?:the\s+)?([a-z0-9_ ]+?)\s+block(?=\s+and|\s*$)", normalized)
    if explicit_pick_match:
        pick_phrase = explicit_pick_match.group(1).strip()
        pick_tokens = sorted(token for token in f"{pick_phrase} block".split() if token)
        for obj in world.objects.values():
            alias_tokens = sorted(token for token in obj.object_id.lower().replace("_", " ").split() if token)
            if pick_tokens == alias_tokens:
                return obj.object_id
        raise ValueError(f"could not resolve pick target '{pick_phrase} block' to a movable/graspable object in world")

    explicit_pick_target = re.search(r"pick\s+(?:the\s+)?([a-z0-9_ ]+?)(?:\s+and|\s*$)", normalized)
    if explicit_pick_target:
        pick_phrase = explicit_pick_target.group(1).strip()
        if pick_phrase:
            for obj in world.objects.values():
                alias_phrase = obj.object_id.lower().replace("_", " ")
                alias_tokens = sorted(token for token in alias_phrase.split() if token)
                pick_tokens = sorted(token for token in pick_phrase.split() if token)
                if pick_phrase == alias_phrase or pick_tokens == alias_tokens:
                    if obj.object_id == TARGET_OBJECT_ID or not (obj.movable or obj.graspable):
                        raise ValueError(
                            f"pick target '{pick_phrase}' is not movable/graspable in the current world"
                        )
                    return obj.object_id
            if pick_phrase in {"target", "target zone", "target_zone"}:
                raise ValueError("pick target 'target' is not movable/graspable in the current world")

    candidates = [
        obj
        for obj in world.objects.values()
        if obj.object_id != TARGET_OBJECT_ID and (obj.movable or obj.graspable)
    ]
    if not candidates:
        raise ValueError("existing world does not contain any movable/graspable source objects")

    scored: list[tuple[int, str]] = []
    for obj in candidates:
        score = 0
        alias_phrase = obj.object_id.lower().replace("_", " ")
        if alias_phrase in normalized:
            score += 100

        metadata = obj.metadata
        color_name = str(metadata.get("color", "")).lower().strip()
        if color_name and color_name in normalized:
            score += 50

        object_tokens = [token for token in re.split(r"[^a-z0-9]+", alias_phrase) if token]
        for token in object_tokens:
            if token in {"obj", "01", "02", "03"}:
                continue
            if token in normalized:
                score += 10

        registry_id = str(metadata.get("registry_id", "")).lower().replace("_", " ")
        if registry_id and registry_id in normalized:
            score += 80

        scored.append((score, obj.object_id))

    scored.sort(key=lambda item: (-item[0], item[1]))
    best_score, best_object_id = scored[0]
    if best_score <= 0:
        if "block_red" in world.objects:
            return "block_red"
        return best_object_id
    return best_object_id


def resolve_target_feature_binding(world: WorldModel, nl: str, source_object_id: str) -> tuple[str, str]:
    normalized = nl.lower().replace("_", " ")
    explicit_surface_match = re.search(r"(?:on|onto|to)\s+the\s+([a-z0-9_ ]+?)\s+surface", normalized)
    if explicit_surface_match:
        target_phrase = explicit_surface_match.group(1).strip()
        if target_phrase == "target":
            return TARGET_OBJECT_ID, "target_surface"
        for object_id in world.objects.keys():
            alias_phrase = object_id.lower().replace("_", " ")
            alias_tokens = sorted(token for token in alias_phrase.split() if token)
            target_tokens = sorted(token for token in target_phrase.split() if token)
            if target_phrase == alias_phrase or target_tokens == alias_tokens:
                return object_id, f"{object_id}_top_surface"
    if "target surface" in normalized:
        return TARGET_OBJECT_ID, "target_surface"
    return TARGET_OBJECT_ID, "target_surface"


def ensure_object_aliases(world: WorldModel, *object_ids: str) -> None:
    for object_id in object_ids:
        if object_id not in world.objects:
            raise ValueError(
                f"existing world does not contain required object alias '{object_id}'. "
                "Automatic alias guessing is disabled."
            )


def ensure_object_frame_and_surface(
    world: WorldModel,
    world_data: dict[str, Any],
    object_id: str,
    *,
    warn_prefix: str = "existing world",
) -> tuple[str, str]:
    obj_payload = world_data["objects"][object_id]
    obj_size = geometry_size(world.get_object(object_id).geometry)
    registry_id = obj_payload.get("metadata", {}).get("registry_id")
    if not registry_id:
        raise ValueError(f"{warn_prefix} must provide metadata.registry_id for '{object_id}'")

    frame_id = f"{object_id}_frame"
    surface_id = f"{object_id}_top_surface"
    frames = world_data.setdefault("frames", {})
    features = world_data.setdefault("features", {})

    if frame_id not in frames:
        frame_payload = {
            "frame_id": frame_id,
            "registry_path": f"{registry_id}/frame",
            "pose": obj_payload["pose"],
            "metadata": {"source": "auto_generated_from_object"},
        }
        print_warn(f"{warn_prefix} is missing '{frame_id}'; generating it from object pose")
        print_auto_fill(frame_id, "object pose", frame_payload)
        frames[frame_id] = frame_payload

    if surface_id not in features:
        feature_payload = {
            "feature_id": surface_id,
            "parent_object": object_id,
            "feature_type": "surface",
            "local_pose": {
                "frame": object_id,
                "position": [0.0, 0.0, obj_size[2] / 2.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
            },
            "size_hint": [obj_size[0], obj_size[1]],
            "metadata": {"source": "auto_generated"},
        }
        print_warn(f"{warn_prefix} is missing '{surface_id}'; generating a top surface feature")
        print_auto_fill(surface_id, f"{object_id} geometry top face", feature_payload)
        features[surface_id] = feature_payload

    return frame_id, surface_id


def ensure_world_features_and_frames(
    world: WorldModel,
    source_object_id: str,
    target_object_id: str | None = None,
    desired_target: Tuple[float, float, float] | None = None,
) -> WorldModel:
    target_object_id = target_object_id or TARGET_OBJECT_ID
    ensure_object_aliases(world, source_object_id, target_object_id)
    world_data = world.model_dump(mode="python")

    block_obj = world.get_object(source_object_id)
    target_obj = world.get_object(target_object_id)
    block_size = geometry_size(block_obj.geometry)
    target_size = geometry_size(target_obj.geometry)
    block_registry_id = block_obj.metadata.get("registry_id")
    target_registry_id = target_obj.metadata.get("registry_id")
    source_frame_id, source_surface_id = ensure_object_frame_and_surface(world, world_data, source_object_id)
    target_frame_id, target_surface_id = ensure_object_frame_and_surface(world, world_data, target_object_id)

    if not block_registry_id or not target_registry_id:
        raise ValueError(
            f"existing world must provide metadata.registry_id for '{source_object_id}' and '{target_object_id}'"
        )

    frames = world_data.setdefault("frames", {})
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
    if START_TCP_FRAME_ID not in frames:
        frame_payload = {
            "frame_id": START_TCP_FRAME_ID,
            "registry_path": "robot/start_tcp_pose",
            "pose": world_data["robot_state"]["tcp_pose"],
            "metadata": {"source": "auto_generated_from_robot_state"},
        }
        print_warn(f"existing world is missing '{START_TCP_FRAME_ID}'; generating it from robot_state.tcp_pose")
        print_auto_fill(START_TCP_FRAME_ID, "robot_state.tcp_pose", frame_payload)
        frames[START_TCP_FRAME_ID] = frame_payload

    features = world_data.setdefault("features", {})
    if target_surface_id == "target_surface" and "target_surface" not in features:
        feature_payload = {
            "feature_id": "target_surface",
            "parent_object": target_object_id,
            "feature_type": "surface",
            "local_pose": {
                "frame": target_object_id,
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

    if desired_target is not None and target_surface_id == "target_surface":
        target_pose = world.get_object(target_object_id).pose
        local_position = [
            desired_target[0] - target_pose.position[0],
            desired_target[1] - target_pose.position[1],
            desired_target[2] - target_pose.position[2],
        ]
        features["target_surface"]["local_pose"] = {
            "frame": target_object_id,
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
    source_object_id = resolve_source_object_alias(world, nl)
    target_object_id, target_feature_id = resolve_target_feature_binding(world, nl, source_object_id)
    source_frame_id = f"{source_object_id}_frame"
    source_feature_id = f"{source_object_id}_top_surface"
    block_registry_id = world.get_object(source_object_id).metadata.get("registry_id")
    target_registry_id = world.get_object(target_object_id).metadata.get("registry_id")
    if not block_registry_id or not target_registry_id:
        raise ValueError(
            f"existing world must provide metadata.registry_id for '{source_object_id}' and '{target_object_id}'"
        )

    now = datetime.now(timezone.utc).isoformat()
    task_id = (
        f"task_pick_place_{uuid.uuid4().hex[:8]}"
        if not task_suffix
        else f"task_{task_suffix}_{uuid.uuid4().hex[:6]}"
    )
    base_frame = world.robot_state.base_frame
    tool_frame = world.robot_state.tcp_pose.frame or base_frame
    source_pose_world = resolve_pose_to_world(world, world.get_object(source_object_id).pose)
    initial_tcp_pose_world = resolve_pose_to_world(world, world.robot_state.tcp_pose)
    target_surface_pose_world = resolve_feature_pose_to_world(world, target_feature_id)
    target_frame_id = f"{target_object_id}_frame"
    target_frame_pose_world = resolve_pose_to_world(world, world.get_frame(target_frame_id).pose)
    source_height = geometry_size(world.get_object(source_object_id).geometry)[2]
    target_height = geometry_size(world.get_object(target_object_id).geometry)[2]
    source_top_surface_offset = source_height / 2.0
    move_linear_z_offset = round(initial_tcp_pose_world["position"][2] - source_pose_world["position"][2], 6)
    final_retreat_distance = round(
        max(0.0, initial_tcp_pose_world["position"][2] - target_surface_pose_world["position"][2]),
        6,
    )
    target_preplace_offset = round((target_height / 2.0) + source_height + PREPLACE_CLEARANCE, 6)
    support_top_z = infer_support_top_z(world, source_object_id)
    pregrasp_offset = source_top_surface_offset + PREGRASP_CLEARANCE
    if support_top_z is not None:
        support_safe_offset = (support_top_z + GRIPPER_HALF_HEIGHT + SUPPORT_CLEARANCE) - source_pose_world["position"][2]
        pregrasp_offset = max(pregrasp_offset, support_safe_offset)
    pregrasp_offset = round(pregrasp_offset, 6)
    transport_tcp_z = source_pose_world["position"][2] + pregrasp_offset + 0.08
    transport_target_offset = round(
        max(target_preplace_offset, transport_tcp_z - target_frame_pose_world["position"][2]),
        6,
    )

    return {
        "ir_version": "0.1",
        "task_id": task_id,
        "created_by": "nl_generator",
        "created_at": now,
        "task_spec": {
            "goal": "pick_and_place",
            "command_text": nl,
            "priority": "normal",
            "success_condition": [f"object_at_destination:{source_object_id}:{target_feature_id}"],
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
                source_object_id: block_registry_id,
                target_object_id: target_registry_id,
            },
            "frames": {
                source_frame_id: world.get_frame(source_frame_id).registry_path,
                target_frame_id: world.get_frame(target_frame_id).registry_path,
                "home_frame": world.get_frame("home_frame").registry_path,
                START_TCP_FRAME_ID: world.get_frame(START_TCP_FRAME_ID).registry_path,
            },
            "regions": {},
            "features": {
                source_feature_id: {
                    "parent_object": source_object_id,
                    "feature_type": "surface",
                    "frame": source_frame_id,
                    "description": "top surface",
                },
                target_feature_id: {
                    "parent_object": target_object_id,
                    "feature_type": "surface",
                    "frame": target_frame_id,
                    "description": "placement surface",
                },
            },
        },
        "action_plan": [
            {"step_id": "s1", "type": "find_object", "inputs": {"object": source_object_id}},
            {
                "step_id": "s2",
                "type": "move_linear",
                "inputs": {
                    "target_pose": {
                        "ref": source_frame_id,
                        "offset": [0.0, 0.0, move_linear_z_offset],
                        "orientation_policy": "align_with_object_top",
                    }
                },
            },
            {
                "step_id": "s3",
                "type": "approach",
                "inputs": {
                    "target_object": source_object_id,
                    "target_feature": source_feature_id,
                    "approach_pose": {
                        "ref": source_frame_id,
                        "offset": [0.0, 0.0, pregrasp_offset],
                        "orientation_policy": "align_with_object_top",
                    },
                },
            },
            {
                "step_id": "s4",
                "type": "grasp",
                "inputs": {
                    "target_object": source_object_id,
                    "target_feature": source_feature_id,
                    "grasp_mode": "pinch",
                },
            },
            {"step_id": "s5", "type": "retreat", "inputs": {"direction": "tool_z_negative", "distance": 0.08}},
            {
                "step_id": "s6",
                "type": "move_linear",
                "inputs": {
                    "target_pose": {
                        "ref": target_frame_id,
                        "offset": [0.0, 0.0, transport_target_offset],
                        "orientation_policy": "keep_current",
                    }
                },
            },
            {
                "step_id": "s7",
                "type": "place",
                "inputs": {
                    "target_object": source_object_id,
                    "target_feature": target_feature_id,
                    "destination_pose": {
                        "ref": target_frame_id,
                        "offset": [0.0, 0.0, transport_target_offset],
                        "orientation_policy": "align_to_target_surface",
                    },
                },
            },
            {"step_id": "s8", "type": "release", "inputs": {"target_object": source_object_id}},
            {"step_id": "s9", "type": "retreat", "inputs": {"direction": "tool_z_negative", "distance": final_retreat_distance}},
            {
                "step_id": "s10",
                "type": "move_linear",
                "inputs": {
                    "target_pose": {
                        "ref": START_TCP_FRAME_ID,
                        "offset": [0.0, 0.0, 0.0],
                        "orientation_policy": "keep_current",
                    }
                },
            },
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


def world_models_equal(left: WorldModel, right: WorldModel) -> bool:
    return left.model_dump(mode="json") == right.model_dump(mode="json")


def save_temp_world_model(world: WorldModel, prefix: str) -> str:
    temp_dir = Path(tempfile.gettempdir())
    temp_path = temp_dir / f"cobot_pipeline_{prefix}_{uuid.uuid4().hex[:8]}.world.json"
    temp_path.write_text(json.dumps(world.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
    print_info(f"Prepared temporary WorldModel JSON: {temp_path}")
    return str(temp_path)


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
    temp_world_path: str | None = None
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
            source_object_id = resolve_source_object_alias(base_world, nl_text)
            target_object_id, _ = resolve_target_feature_binding(base_world, nl_text, source_object_id)
            world_model = ensure_world_features_and_frames(
                base_world,
                source_object_id=source_object_id,
                target_object_id=target_object_id,
                desired_target=desired_target,
            )
        else:
            world_model = base_world

        ir_data = build_ir_from_world(nl_text, world_model)
        ir_model = validate_ir_dict(ir_data)

        world_path: str | None = None
        if args.existing_world:
            if world_models_equal(base_world, world_model):
                world_path = str(Path(args.existing_world))
                print_info(f"Reusing existing world file without writing a derived copy: {world_path}")
            elif args.no_run:
                world_path = save_world_model(world_model, world_sample_path(prefix))
                print_info("Saved derived world because generated IR depends on world adjustments")
            else:
                temp_world_path = save_temp_world_model(world_model, prefix)
                world_path = temp_world_path
                print_info("Using a temporary derived world for this run; no sample world file will be created")
        else:
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
        finally:
            if temp_world_path is not None:
                try:
                    Path(temp_world_path).unlink(missing_ok=True)
                except OSError as cleanup_error:
                    print_warn(f"failed to remove temporary world file '{temp_world_path}': {cleanup_error}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
