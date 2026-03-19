import json
from pathlib import Path

from executor import run_ir
from ir_models import GenericCobotIR
from run_demo import validate_ir_world_consistency
from world_model import WorldModel


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_sample_pick_place_with_world():
    ir = GenericCobotIR.model_validate(load_json("samples/sample_pick_place.json"))
    world = WorldModel.model_validate(load_json("samples/sample_pick_place.world.json"))

    errors, warnings = validate_ir_world_consistency(ir, world)
    result = run_ir(ir, world_model=world)

    assert errors == []
    assert warnings == []
    assert result.status == "passed"


def test_ir_only_runs():
    ir = GenericCobotIR.model_validate(load_json("samples/sample_pick_place.json"))
    result = run_ir(ir, world_model=None)
    assert result is not None


def test_registry_id_missing_emits_warning():
    ir_data = load_json("samples/sample_pick_place.json")
    world_data = load_json("samples/sample_pick_place.world.json")
    del world_data["objects"]["block_red"]["metadata"]["registry_id"]

    ir = GenericCobotIR.model_validate(ir_data)
    world = WorldModel.model_validate(world_data)
    errors, warnings = validate_ir_world_consistency(ir, world)

    assert errors == []
    assert any("missing metadata.registry_id" in warning for warning in warnings)


def test_registry_id_mismatch_is_error():
    ir_data = load_json("samples/sample_pick_place.json")
    world_data = load_json("samples/sample_pick_place.world.json")
    world_data["objects"]["block_red"]["metadata"]["registry_id"] = "obj_block_red_wrong"

    ir = GenericCobotIR.model_validate(ir_data)
    world = WorldModel.model_validate(world_data)
    errors, warnings = validate_ir_world_consistency(ir, world)

    assert any("registry_id mismatch" in error for error in errors)
    assert warnings == []


def test_move_linear_aabb_collision_detected():
    ir = GenericCobotIR.model_validate(load_json("samples/sample_pick_place.json"))
    world_data = load_json("samples/sample_pick_place.world.json")
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
    assert result.errors[0].error_id == "e_geometry_collision"
    assert result.errors[0].step_id == "s5"


def test_insert_geometry_collision_detected():
    ir = GenericCobotIR.model_validate(load_json("samples/sample_insert.json"))
    world_data = load_json("samples/sample_insert.world.json")
    world_data["features"]["hole"]["width"] = 0.006
    world = WorldModel.model_validate(world_data)

    result = run_ir(ir, world_model=world)

    assert result.status == "failed"
    assert result.errors[0].error_id == "e_insert_geometry_collision"
    assert result.errors[0].step_id == "s8"
