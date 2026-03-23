import json
from importlib.util import find_spec
from pathlib import Path

import pytest

from executor import run_ir
from ir_models import GenericCobotIR
from run_demo import resolve_simulator_backend, validate_ir_world_consistency
from sample_paths import ir_sample_path, world_sample_path
from world_model import WorldModel


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_sample_pick_place_with_world():
    ir = GenericCobotIR.model_validate(load_json(str(ir_sample_path("sample_pick_place"))))
    world = WorldModel.model_validate(load_json(str(world_sample_path("sample_pick_place"))))

    errors, warnings = validate_ir_world_consistency(ir, world)
    result = run_ir(ir, world_model=world)

    assert errors == []
    assert warnings == []
    assert result.status == "passed"


def test_ir_only_runs():
    ir = GenericCobotIR.model_validate(load_json(str(ir_sample_path("sample_pick_place"))))
    result = run_ir(ir, world_model=None)
    assert result is not None


def test_registry_id_missing_emits_warning():
    ir_data = load_json(str(ir_sample_path("sample_pick_place")))
    world_data = load_json(str(world_sample_path("sample_pick_place")))
    del world_data["objects"]["block_red"]["metadata"]["registry_id"]

    ir = GenericCobotIR.model_validate(ir_data)
    world = WorldModel.model_validate(world_data)
    errors, warnings = validate_ir_world_consistency(ir, world)

    assert errors == []
    assert any("missing metadata.registry_id" in warning for warning in warnings)


def test_registry_id_mismatch_is_error():
    ir_data = load_json(str(ir_sample_path("sample_pick_place")))
    world_data = load_json(str(world_sample_path("sample_pick_place")))
    world_data["objects"]["block_red"]["metadata"]["registry_id"] = "obj_block_red_wrong"

    ir = GenericCobotIR.model_validate(ir_data)
    world = WorldModel.model_validate(world_data)
    errors, warnings = validate_ir_world_consistency(ir, world)

    assert any("registry_id mismatch" in error for error in errors)
    assert warnings == []


def test_move_linear_aabb_collision_detected():
    ir = GenericCobotIR.model_validate(load_json(str(ir_sample_path("sample_pick_place"))))
    world_data = load_json(str(world_sample_path("sample_pick_place")))
    world_data["objects"]["blocking_obstacle"] = {
        "object_id": "blocking_obstacle",
        "object_type": "obstacle",
        "pose": {
            "frame": "world",
            "position": [0.55, 0.12, 0.11],
            "orientation": [0.0, 0.0, 0.0, 1.0],
        },
        "geometry": {
            "type": "box",
            "size": [0.05, 0.05, 0.05],
        },
        "movable": False,
        "graspable": False,
        "collision_enabled": True,
        "metadata": {
            "registry_id": "obj_blocking_obstacle_01",
        },
    }
    world = WorldModel.model_validate(world_data)

    result = run_ir(ir, world_model=world)

    assert result.status == "failed"
    assert result.errors[0].error_id == "e_aabb_collision"
    assert result.errors[0].step_id == "s5"


def test_insert_geometry_collision_detected():
    ir = GenericCobotIR.model_validate(load_json(str(ir_sample_path("sample_insert"))))
    world_data = load_json(str(world_sample_path("sample_insert")))
    world_data["features"]["hole"]["width"] = 0.006
    world = WorldModel.model_validate(world_data)

    result = run_ir(ir, world_model=world)

    assert result.status == "failed"
    assert result.errors[0].error_id == "e_insert_geometry_collision"
    assert result.errors[0].step_id == "s8"


def test_unknown_simulator_backend_raises():
    ir = GenericCobotIR.model_validate(load_json(str(ir_sample_path("sample_pick_place"))))

    with pytest.raises(ValueError, match="unknown simulator_backend"):
        run_ir(ir, simulator_backend="invalid_backend")


def test_pybullet_backend_requires_pybullet_or_world():
    ir = GenericCobotIR.model_validate(load_json(str(ir_sample_path("sample_pick_place"))))
    world = WorldModel.model_validate(load_json(str(world_sample_path("sample_pick_place"))))

    if find_spec("pybullet") is None:
        with pytest.raises(RuntimeError, match="pybullet is not installed"):
            run_ir(ir, world_model=world, simulator_backend="pybullet")
    else:
        result = run_ir(ir, world_model=world, simulator_backend="pybullet")
        assert result.simulator == "pybullet_sim"


def test_resolve_simulator_backend_falls_back_to_mock_without_world():
    assert resolve_simulator_backend("pybullet", None) == "mock"


def test_resolve_simulator_backend_keeps_pybullet_with_world():
    world = WorldModel.model_validate(load_json(str(world_sample_path("sample_pick_place"))))
    assert resolve_simulator_backend("pybullet", world) == "pybullet"
