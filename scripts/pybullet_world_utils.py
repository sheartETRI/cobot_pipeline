"""Helpers for building and exporting a PyBullet scene as a WorldModel."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional, Tuple

from pydantic import ValidationError

from world_model import WorldModel

try:
    import pybullet as p
except ImportError:  # pragma: no cover - exercised in environments without pybullet
    p = None


logger = logging.getLogger(__name__)

DEFAULT_ORIENTATION = [0.0, 0.0, 0.0, 1.0]
DEFAULT_TCP_POSE = {
    "frame": "world",
    "position": [0.4, 0.0, 0.3],
    "orientation": [0.0, 0.0, 0.0, 1.0],
}

registry_map: dict[str, int] = {}
body_info_map: dict[int, dict[str, Any]] = {}
feature_map: dict[str, dict[str, Any]] = {}
current_client_id: int | None = None


def _require_pybullet() -> None:
    if p is None:
        raise RuntimeError("pybullet is not installed. Install it with 'pip install pybullet'.")


def _current_client() -> int:
    if current_client_id is not None:
        return current_client_id
    return 0


def _normalize_orientation(orientation: Optional[list[float]]) -> list[float]:
    return list(orientation) if orientation is not None else list(DEFAULT_ORIENTATION)


def _sanitize_registry_as_object_id(registry_id: str) -> str:
    if registry_id.startswith("obj_"):
        match = re.match(r"obj_(.+?)_\d+$", registry_id)
        if match:
            return match.group(1)
        return registry_id[4:]
    return registry_id.replace("/", "_")


def _register_body(
    body_id: int,
    registry_id: str,
    *,
    object_id: Optional[str] = None,
    object_type: str = "object",
    size: Tuple[float, float, float] = (0.04, 0.04, 0.04),
    collision_enabled: bool = True,
    graspable: bool = True,
    movable: bool = True,
    color: Tuple[float, float, float, float] | None = None,
) -> None:
    registry_map[registry_id] = body_id
    body_info_map[body_id] = {
        "registry_id": registry_id,
        "object_id": object_id or _sanitize_registry_as_object_id(registry_id),
        "object_type": object_type,
        "size": list(size),
        "collision_enabled": collision_enabled,
        "graspable": graspable,
        "movable": movable,
        "color": list(color) if color is not None else None,
    }


def connect(gui: bool = True) -> int:
    """Connect to PyBullet in GUI or DIRECT mode."""
    _require_pybullet()
    global current_client_id
    if p.isConnected():
        current_client_id = _current_client()
        return current_client_id

    mode = p.GUI if gui else p.DIRECT
    client_id = p.connect(mode)
    if client_id < 0:
        raise RuntimeError(f"failed to connect to pybullet in mode={'GUI' if gui else 'DIRECT'}")
    p.setGravity(0.0, 0.0, -9.81, physicsClientId=client_id)
    current_client_id = client_id
    return client_id


def disconnect() -> None:
    """Disconnect the active PyBullet client and clear registries."""
    global current_client_id
    _require_pybullet()
    if p.isConnected():
        p.disconnect(physicsClientId=_current_client())
    current_client_id = None
    registry_map.clear()
    body_info_map.clear()
    feature_map.clear()


def create_box(
    registry_id: str,
    position: Tuple[float, float, float],
    orientation: list[float] = DEFAULT_ORIENTATION,
    size: Tuple[float, float, float] = (0.04, 0.04, 0.04),
    mass: float = 0.2,
    color: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
    collision_enabled: bool = True,
    graspable: bool = True,
    movable: bool = True,
) -> int:
    """Create a box body and register it by registry_id."""
    _require_pybullet()
    if not p.isConnected():
        raise RuntimeError("pybullet is not connected. Call connect() before create_box().")
    client_id = _current_client()
    half_extents = [axis / 2.0 for axis in size]
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=client_id)
    visual_shape = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=list(color),
        physicsClientId=client_id,
    )
    body_id = p.createMultiBody(
        baseMass=mass if movable else 0.0,
        baseCollisionShapeIndex=collision_shape if collision_enabled else -1,
        baseVisualShapeIndex=visual_shape,
        basePosition=list(position),
        baseOrientation=_normalize_orientation(orientation),
        physicsClientId=client_id,
    )
    _register_body(
        body_id,
        registry_id,
        object_type="block" if movable else "fixture",
        size=size,
        collision_enabled=collision_enabled,
        graspable=graspable,
        movable=movable,
        color=color,
    )
    return body_id


def create_fixture_box(
    registry_id: str,
    position: Tuple[float, float, float],
    orientation: list[float] = DEFAULT_ORIENTATION,
    size: Tuple[float, float, float] = (0.2, 0.15, 0.02),
    mass: float = 0.0,
    color: Tuple[float, float, float, float] = (0.3, 0.3, 0.3, 1.0),
    collision_enabled: bool = True,
    graspable: bool = False,
    movable: bool = False,
) -> int:
    """Convenience wrapper for a static fixture box."""
    return create_box(
        registry_id=registry_id,
        position=position,
        orientation=orientation,
        size=size,
        mass=mass,
        color=color,
        collision_enabled=collision_enabled,
        graspable=graspable,
        movable=movable,
    )


def add_feature(
    feature_id: str,
    parent_registry_id: str,
    local_position: Tuple[float, float, float],
    local_orientation: list[float] = DEFAULT_ORIENTATION,
    size_hint: Optional[list[float]] = None,
) -> None:
    """Register a logical feature on top of a parent body."""
    if parent_registry_id not in registry_map:
        raise ValueError(f"parent registry_id '{parent_registry_id}' is not registered")
    feature_map[feature_id] = {
        "parent": parent_registry_id,
        "local_pose": {
            "position": list(local_position),
            "orientation": _normalize_orientation(local_orientation),
        },
        "size_hint": list(size_hint) if size_hint is not None else None,
    }


def _ensure_export_connection() -> tuple[int, bool]:
    _require_pybullet()
    global current_client_id
    if p.isConnected():
        current_client_id = _current_client()
        return current_client_id, False

    shared_client = p.connect(p.SHARED_MEMORY)
    if shared_client >= 0:
        current_client_id = shared_client
        return shared_client, True

    raise RuntimeError(
        "PyBullet is not connected in this process and no shared-memory session is available. "
        "Create the scene in this process or start a shared-memory PyBullet server."
    )


def _infer_geometry(body_id: int) -> dict[str, Any]:
    info = body_info_map.get(body_id, {})
    size = info.get("size")
    client_id = _current_client()
    visual_data = p.getVisualShapeData(body_id, physicsClientId=client_id)
    if size is not None:
        return {"type": "box", "size": list(size)}
    if not visual_data:
        return {"type": "box", "size": [0.04, 0.04, 0.04]}

    visual = visual_data[0]
    geom_type = visual[2]
    dimensions = visual[3]
    if geom_type == p.GEOM_BOX:
        if len(dimensions) >= 3:
            return {"type": "box", "size": [float(dimensions[0]) * 2.0, float(dimensions[1]) * 2.0, float(dimensions[2]) * 2.0]}
        return {"type": "box", "size": [0.04, 0.04, 0.04]}
    if geom_type == p.GEOM_CYLINDER:
        radius = float(dimensions[0]) if len(dimensions) >= 1 else 0.02
        height = float(dimensions[1]) if len(dimensions) >= 2 else 0.04
        return {"type": "cylinder", "radius": radius, "height": height}
    if geom_type == p.GEOM_PLANE:
        size_xy = [1.0, 1.0]
        if len(dimensions) >= 2:
            size_xy = [float(dimensions[0]), float(dimensions[1])]
        return {"type": "plane", "size": size_xy}
    return {"type": "box", "size": [0.04, 0.04, 0.04]}


def _base_pose(body_id: int, world_frame: str) -> dict[str, Any]:
    client_id = _current_client()
    position, orientation = p.getBasePositionAndOrientation(body_id, physicsClientId=client_id)
    return {
        "frame": world_frame,
        "position": list(position),
        "orientation": list(orientation),
    }


def _select_robot_body(body_ids: list[int]) -> int | None:
    client_id = _current_client()
    robot_body = None
    max_joints = 0
    for body_id in body_ids:
        joint_count = p.getNumJoints(body_id, physicsClientId=client_id)
        if joint_count > max_joints:
            max_joints = joint_count
            robot_body = body_id
    return robot_body


def _infer_tcp_pose(robot_body: int | None, world_frame: str) -> dict[str, Any]:
    client_id = _current_client()
    if robot_body is None:
        return dict(DEFAULT_TCP_POSE)

    joint_count = p.getNumJoints(robot_body, physicsClientId=client_id)
    chosen_link = None
    chosen_name = None
    for joint_index in range(joint_count):
        joint_info = p.getJointInfo(robot_body, joint_index, physicsClientId=client_id)
        link_name = joint_info[12].decode("utf-8")
        if any(token in link_name.lower() for token in ("tool", "tcp", "ee", "gripper")):
            chosen_link = joint_index
            chosen_name = link_name
            break
        chosen_link = joint_index
        chosen_name = link_name

    if chosen_link is None:
        return _base_pose(robot_body, world_frame)

    link_state = p.getLinkState(robot_body, chosen_link, physicsClientId=client_id)
    logger.debug("Selected TCP link %s for robot body %s", chosen_name, robot_body)
    return {
        "frame": world_frame,
        "position": list(link_state[4]),
        "orientation": list(link_state[5]),
    }


def export_world_model(path: str = "pybullet_exported_world.json", world_frame: str = "world") -> str:
    """Export the current PyBullet state to a validated WorldModel JSON file."""
    client_id, temporary_shared_connection = _ensure_export_connection()
    if not p.isConnected(physicsClientId=client_id):
        raise RuntimeError("PyBullet connection is not active")

    body_ids = [p.getBodyUniqueId(index, physicsClientId=client_id) for index in range(p.getNumBodies(physicsClientId=client_id))]
    objects: dict[str, Any] = {}
    frames: dict[str, Any] = {}
    registry_to_object_id: dict[str, str] = {}

    for body_id in body_ids:
        info = body_info_map.get(body_id)
        if info is None:
            registry_id = f"auto_obj_{body_id}"
            logger.warning("body_id=%s is missing registry_id; using %s", body_id, registry_id)
            print(f"[WARN] body_id={body_id} has no registry_id; using '{registry_id}'")
            info = {
                "registry_id": registry_id,
                "object_id": registry_id,
                "object_type": "object",
                "collision_enabled": True,
                "graspable": True,
                "movable": True,
                "color": None,
            }
            body_info_map[body_id] = info
            registry_map[registry_id] = body_id

        registry_id = info["registry_id"]
        object_id = info.get("object_id") or _sanitize_registry_as_object_id(registry_id)
        registry_to_object_id[registry_id] = object_id

        dynamics = p.getDynamicsInfo(body_id, -1, physicsClientId=client_id)
        pose = _base_pose(body_id, world_frame)
        movable = info.get("movable", float(dynamics[0]) > 0.0)
        graspable = info.get("graspable", movable)
        collision_enabled = info.get("collision_enabled", True)
        object_type = info.get("object_type", "fixture" if not movable else "object")

        metadata = {"registry_id": registry_id}
        if info.get("color") is not None:
            metadata["rgba"] = ",".join(str(value) for value in info["color"])

        objects[object_id] = {
            "object_id": object_id,
            "object_type": object_type,
            "pose": pose,
            "geometry": _infer_geometry(body_id),
            "movable": movable,
            "graspable": graspable,
            "collision_enabled": collision_enabled,
            "metadata": metadata,
        }
        frames[f"{object_id}_frame"] = {
            "frame_id": f"{object_id}_frame",
            "registry_path": f"{registry_id}/frame",
            "pose": pose,
            "metadata": {"source": "pybullet_export"},
        }

    features: dict[str, Any] = {}
    for feature_id, feature in feature_map.items():
        parent_registry_id = feature["parent"]
        parent_object = registry_to_object_id.get(parent_registry_id)
        if parent_object is None:
            logger.warning("Skipping feature '%s' because parent '%s' is unknown", feature_id, parent_registry_id)
            continue
        features[feature_id] = {
            "feature_id": feature_id,
            "parent_object": parent_object,
            "feature_type": "surface",
            "local_pose": {
                "frame": parent_object,
                "position": list(feature["local_pose"]["position"]),
                "orientation": list(feature["local_pose"]["orientation"]),
            },
            "size_hint": feature["size_hint"],
            "metadata": {"source": "pybullet_export"},
        }

    robot_body = _select_robot_body(body_ids)
    joint_positions: list[float] = []
    if robot_body is not None:
        for joint_index in range(p.getNumJoints(robot_body, physicsClientId=client_id)):
            joint_state = p.getJointState(robot_body, joint_index, physicsClientId=client_id)
            joint_positions.append(float(joint_state[0]))
    tcp_pose = _infer_tcp_pose(robot_body, world_frame)
    frames["home_frame"] = {
        "frame_id": "home_frame",
        "registry_path": "robot/home_pose",
        "pose": tcp_pose,
        "metadata": {"source": "pybullet_export"},
    }

    world_data = {
        "world_frame": world_frame,
        "objects": objects,
        "features": features,
        "frames": frames,
        "robot_state": {
            "base_frame": world_frame,
            "tcp_pose": tcp_pose,
            "joint_positions": joint_positions,
            "attached_object": None,
        },
    }

    try:
        world_model = WorldModel.model_validate(world_data)
    except ValidationError as error:
        raise ValueError(f"exported world failed validation: {error}") from error

    output_path = Path(path)
    output_path.write_text(
        json.dumps(world_model.model_dump(mode="python"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if temporary_shared_connection:
        p.disconnect(physicsClientId=client_id)

    logger.info(
        "Exported world model path=%s objects=%d features=%d frames=%d",
        output_path,
        len(world_model.objects),
        len(world_model.features),
        len(world_model.frames),
    )
    print(
        "[INFO] Exported WorldModel JSON "
        f"({output_path}) objects={len(world_model.objects)} "
        f"features={len(world_model.features)} frames={len(world_model.frames)}"
    )
    return str(output_path)
