#!/usr/bin/env python3
"""Build a WorldModel JSON from concise CLI declarations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sample_paths import ensure_sample_dirs, world_sample_path
from world_model import WorldModel

DEFAULT_ORIENTATION = [0.0, 0.0, 0.0, 1.0]
DEFAULT_TCP_POSITION = [0.4, 0.0, 0.3]
DEFAULT_JOINT_POSITIONS = [0.0, -1.2, 1.5, 0.0, 1.2, 0.0]
DEFAULT_BLOCK_SIZE = [0.04, 0.04, 0.04]
DEFAULT_TARGET_SIZE = [0.2, 0.15, 0.02]
COLOR_RGBA = {
    "red": "1.0,0.0,0.0,1.0",
    "blue": "0.0,0.0,1.0,1.0",
    "green": "0.0,1.0,0.0,1.0",
    "yellow": "1.0,1.0,0.0,1.0",
    "gray": "0.3,0.3,0.3,1.0",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a WorldModel JSON from compact CLI declarations")
    parser.add_argument("--yaml", type=str, default=None, help="Optional YAML spec file with blocks/fixtures/tcp")
    parser.add_argument("--block", action="append", default=[], help="Block spec: <name>@x,y,z or <object_id>@x,y,z")
    parser.add_argument(
        "--fixture",
        action="append",
        default=[],
        help="Fixture spec: <name>@x,y,z or <name>@x,y,z@sx,sy,sz",
    )
    parser.add_argument("--tcp", type=str, default="0.4,0.0,0.3", help="Initial TCP position x,y,z")
    parser.add_argument("--out", type=str, default=None, help="Output world JSON path")
    return parser.parse_args(argv)


def parse_xyz(text: str) -> list[float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected x,y,z triple, got '{text}'")
    return [float(part) for part in parts]


def parse_size(text: str) -> list[float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected sx,sy,sz triple, got '{text}'")
    size = [float(part) for part in parts]
    if any(value <= 0.0 for value in size):
        raise ValueError(f"size values must be positive: '{text}'")
    return size


def normalize_block_id(name: str) -> tuple[str, str]:
    base_name = name.strip()
    if not base_name:
        raise ValueError("block name must not be empty")
    if base_name.startswith("block_"):
        object_id = base_name
        color_name = base_name.removeprefix("block_")
    else:
        object_id = f"block_{base_name}"
        color_name = base_name
    return object_id, color_name


def normalize_fixture_id(name: str) -> str:
    base_name = name.strip()
    if not base_name:
        raise ValueError("fixture name must not be empty")
    if base_name == "target":
        return "target_zone"
    return base_name


def parse_block_spec(spec: str) -> dict[str, object]:
    try:
        name, position_text = spec.split("@", 1)
    except ValueError as error:
        raise ValueError(f"invalid --block spec '{spec}', expected <name>@x,y,z") from error
    object_id, color_name = normalize_block_id(name)
    return {
        "object_id": object_id,
        "color_name": color_name,
        "position": parse_xyz(position_text),
    }


def parse_fixture_spec(spec: str) -> dict[str, object]:
    parts = spec.split("@")
    if len(parts) not in {2, 3}:
        raise ValueError(f"invalid --fixture spec '{spec}', expected <name>@x,y,z or <name>@x,y,z@sx,sy,sz")
    name = normalize_fixture_id(parts[0])
    position = parse_xyz(parts[1])
    size = parse_size(parts[2]) if len(parts) == 3 else list(DEFAULT_TARGET_SIZE)
    return {
        "object_id": name,
        "position": position,
        "size": size,
    }


def object_frame(object_id: str, registry_id: str, position: list[float]) -> dict[str, object]:
    return {
        "frame_id": f"{object_id}_frame",
        "registry_path": f"{registry_id}/frame",
        "pose": {
            "frame": "world",
            "position": list(position),
            "orientation": list(DEFAULT_ORIENTATION),
        },
        "metadata": {"source": "make_world"},
    }


def top_surface_feature(feature_id: str, parent_object: str, half_height: float, size_hint: list[float], usage: str) -> dict[str, object]:
    return {
        "feature_id": feature_id,
        "parent_object": parent_object,
        "feature_type": "surface",
        "local_pose": {
            "frame": parent_object,
            "position": [0.0, 0.0, half_height],
            "orientation": list(DEFAULT_ORIENTATION),
        },
        "size_hint": list(size_hint),
        "metadata": {"usage": usage},
    }


def load_yaml_spec(path: str) -> dict[str, object]:
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"yaml file not found: {yaml_path}")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("yaml root must be a mapping")
    return data


def yaml_block_to_spec(entry: object) -> str:
    if not isinstance(entry, dict):
        raise ValueError("each yaml blocks entry must be a mapping")
    name = entry.get("name") or entry.get("id")
    position = entry.get("position")
    if name is None or position is None:
        raise ValueError("yaml block entry requires 'name' (or 'id') and 'position'")
    if not isinstance(position, list) or len(position) != 3:
        raise ValueError("yaml block position must be a 3-element list")
    return f"{name}@{position[0]},{position[1]},{position[2]}"


def yaml_fixture_to_spec(entry: object) -> str:
    if not isinstance(entry, dict):
        raise ValueError("each yaml fixtures entry must be a mapping")
    name = entry.get("name") or entry.get("id")
    position = entry.get("position")
    size = entry.get("size")
    if name is None or position is None:
        raise ValueError("yaml fixture entry requires 'name' (or 'id') and 'position'")
    if not isinstance(position, list) or len(position) != 3:
        raise ValueError("yaml fixture position must be a 3-element list")
    spec = f"{name}@{position[0]},{position[1]},{position[2]}"
    if size is not None:
        if not isinstance(size, list) or len(size) != 3:
            raise ValueError("yaml fixture size must be a 3-element list")
        spec = f"{spec}@{size[0]},{size[1]},{size[2]}"
    return spec


def collect_specs(args: argparse.Namespace) -> tuple[list[str], list[str], str]:
    block_specs = list(args.block)
    fixture_specs = list(args.fixture)
    tcp_spec = args.tcp

    if args.yaml:
        yaml_data = load_yaml_spec(args.yaml)
        yaml_blocks = yaml_data.get("blocks", [])
        yaml_fixtures = yaml_data.get("fixtures", [])
        yaml_tcp = yaml_data.get("tcp")

        if yaml_blocks is not None:
            if not isinstance(yaml_blocks, list):
                raise ValueError("yaml 'blocks' must be a list")
            block_specs = [yaml_block_to_spec(entry) for entry in yaml_blocks] + block_specs
        if yaml_fixtures is not None:
            if not isinstance(yaml_fixtures, list):
                raise ValueError("yaml 'fixtures' must be a list")
            fixture_specs = [yaml_fixture_to_spec(entry) for entry in yaml_fixtures] + fixture_specs
        if yaml_tcp is not None:
            if not isinstance(yaml_tcp, list) or len(yaml_tcp) != 3:
                raise ValueError("yaml 'tcp' must be a 3-element list")
            tcp_spec = f"{yaml_tcp[0]},{yaml_tcp[1]},{yaml_tcp[2]}"

    return block_specs, fixture_specs, tcp_spec


def build_world(args: argparse.Namespace) -> WorldModel:
    block_specs, fixture_specs, tcp_spec = collect_specs(args)
    if not block_specs and not fixture_specs:
        raise ValueError("at least one --block or --fixture must be provided")

    objects: dict[str, object] = {}
    features: dict[str, object] = {}
    frames: dict[str, object] = {}

    for raw_spec in block_specs:
        block = parse_block_spec(raw_spec)
        object_id = str(block["object_id"])
        if object_id in objects:
            raise ValueError(f"duplicate object_id '{object_id}'")
        position = list(block["position"])
        color_name = str(block["color_name"])
        registry_id = f"obj_{object_id}_01"
        metadata = {
            "registry_id": registry_id,
            "color": color_name,
        }
        if color_name in COLOR_RGBA:
            metadata["rgba"] = COLOR_RGBA[color_name]
        objects[object_id] = {
            "object_id": object_id,
            "object_type": "block",
            "pose": {
                "frame": "world",
                "position": position,
                "orientation": list(DEFAULT_ORIENTATION),
            },
            "geometry": {"type": "box", "size": list(DEFAULT_BLOCK_SIZE)},
            "movable": True,
            "graspable": True,
            "collision_enabled": True,
            "metadata": metadata,
        }
        frames[f"{object_id}_frame"] = object_frame(object_id, registry_id, position)
        features[f"{object_id}_top_surface"] = top_surface_feature(
            f"{object_id}_top_surface",
            object_id,
            DEFAULT_BLOCK_SIZE[2] / 2.0,
            DEFAULT_BLOCK_SIZE[:2],
            "grasp_reference_surface",
        )

    for raw_spec in fixture_specs:
        fixture = parse_fixture_spec(raw_spec)
        object_id = str(fixture["object_id"])
        if object_id in objects:
            raise ValueError(f"duplicate object_id '{object_id}'")
        position = list(fixture["position"])
        size = list(fixture["size"])
        registry_id = f"obj_{object_id}_01"
        metadata = {
            "registry_id": registry_id,
            "role": "placement_fixture",
        }
        if object_id == "target_zone":
            metadata["rgba"] = COLOR_RGBA["gray"]
        objects[object_id] = {
            "object_id": object_id,
            "object_type": "fixture",
            "pose": {
                "frame": "world",
                "position": position,
                "orientation": list(DEFAULT_ORIENTATION),
            },
            "geometry": {"type": "box", "size": size},
            "movable": False,
            "graspable": False,
            "collision_enabled": True,
            "metadata": metadata,
        }
        frames[f"{object_id}_frame"] = object_frame(object_id, registry_id, position)
        feature_id = "target_surface" if object_id == "target_zone" else f"{object_id}_top_surface"
        usage = "flat_placement_surface" if object_id == "target_zone" else "support_surface"
        features[feature_id] = top_surface_feature(feature_id, object_id, size[2] / 2.0, size[:2], usage)

    tcp_position = parse_xyz(tcp_spec)
    frames["home_frame"] = {
        "frame_id": "home_frame",
        "registry_path": "robot/home_pose",
        "pose": {
            "frame": "world",
            "position": tcp_position,
            "orientation": list(DEFAULT_ORIENTATION),
        },
        "metadata": {"source": "make_world"},
    }

    world_data = {
        "world_frame": "world",
        "objects": objects,
        "features": features,
        "frames": frames,
        "robot_state": {
            "base_frame": "world",
            "tcp_pose": {
                "frame": "world",
                "position": tcp_position,
                "orientation": list(DEFAULT_ORIENTATION),
            },
            "joint_positions": list(DEFAULT_JOINT_POSITIONS),
            "attached_object": None,
        },
    }
    return WorldModel.model_validate(world_data)


def save_world(world: WorldModel, output_path: Path) -> Path:
    ensure_sample_dirs()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(world.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    world = build_world(args)
    output_path = Path(args.out) if args.out else world_sample_path("generated_world")
    saved_path = save_world(world, output_path)
    print(
        f"[INFO] Saved WorldModel JSON: {saved_path} "
        f"(objects={len(world.objects)} features={len(world.features)} frames={len(world.frames)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
