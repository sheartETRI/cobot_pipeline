import json
from importlib.util import find_spec
from pathlib import Path

import pytest

from executor import run_ir, run_ir_with_final_world
from ir_models import GenericCobotIR
from run_demo import resolve_simulator_backend, validate_ir_world_consistency
from sample_paths import ir_sample_path, world_sample_path
from scripts import generate_and_run_rule as generator
from scripts import interactive_session
from scripts import make_world
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


def test_build_ir_from_world_selects_blue_block_from_nl():
    world = WorldModel.model_validate(load_json(str(world_sample_path("demo_multi_blocks"))))
    world = generator.ensure_world_features_and_frames(world, source_object_id="block_blue")

    ir_data = generator.build_ir_from_world("pick the blue block and place it on the target surface", world)
    ir = GenericCobotIR.model_validate(ir_data)

    assert ir.action_plan[0].inputs["object"] == "block_blue"
    assert ir.action_plan[1].type == "move_linear"
    assert ir.action_plan[1].inputs["target_pose"]["ref"] == "block_blue_frame"
    assert ir.action_plan[1].inputs["target_pose"]["offset"] == [0.0, 0.0, 0.28]
    assert ir.action_plan[2].inputs["target_object"] == "block_blue"
    assert ir.action_plan[2].inputs["target_feature"] == "block_blue_top_surface"
    assert ir.world_binding.frames["start_tcp_frame"] == "robot/start_tcp_pose"
    assert ir.action_plan[8].inputs["distance"] == 0.28
    assert ir.action_plan[9].inputs["target_pose"]["ref"] == "start_tcp_frame"
    assert ir.task_spec.success_condition == ["object_at_destination:block_blue:target_surface"]


def test_build_ir_from_world_selects_green_block_from_nl():
    world = WorldModel.model_validate(load_json(str(world_sample_path("demo_multi_blocks"))))
    world = generator.ensure_world_features_and_frames(world, source_object_id="block_green")

    ir_data = generator.build_ir_from_world("pick the green block and place it on the target surface", world)
    ir = GenericCobotIR.model_validate(ir_data)

    assert ir.action_plan[0].inputs["object"] == "block_green"
    assert ir.action_plan[1].type == "move_linear"
    assert ir.action_plan[1].inputs["target_pose"]["ref"] == "block_green_frame"
    assert ir.action_plan[1].inputs["target_pose"]["offset"] == [0.0, 0.0, 0.28]
    assert ir.action_plan[2].inputs["target_object"] == "block_green"
    assert ir.action_plan[2].inputs["target_feature"] == "block_green_top_surface"
    assert ir.world_binding.frames["start_tcp_frame"] == "robot/start_tcp_pose"
    assert ir.action_plan[8].inputs["distance"] == 0.28
    assert ir.action_plan[9].inputs["target_pose"]["ref"] == "start_tcp_frame"
    assert ir.task_spec.success_condition == ["object_at_destination:block_green:target_surface"]


def test_build_ir_from_world_supports_stacking_destination():
    world = WorldModel.model_validate(load_json(str(world_sample_path("demo_multi_blocks"))))
    world = generator.ensure_world_features_and_frames(
        world,
        source_object_id="block_blue",
        target_object_id="block_red",
    )

    ir_data = generator.build_ir_from_world("pick the blue block and place it on the red block surface", world)
    ir = GenericCobotIR.model_validate(ir_data)

    assert ir.world_binding.frames["block_red_frame"] == "obj_block_red_01/frame"
    assert ir.world_binding.features["block_red_top_surface"].parent_object == "block_red"
    assert ir.action_plan[5].inputs["target_pose"]["ref"] == "block_red_frame"
    assert ir.action_plan[6].inputs["target_feature"] == "block_red_top_surface"
    assert ir.action_plan[5].inputs["target_pose"]["offset"] == [0.0, 0.0, 0.12]
    assert ir.task_spec.success_condition == ["object_at_destination:block_blue:block_red_top_surface"]


def test_build_ir_from_world_rejects_non_graspable_pick_target():
    world = WorldModel.model_validate(load_json(str(world_sample_path("demo_multi_blocks"))))
    world = generator.ensure_world_features_and_frames(world, source_object_id="block_green", target_object_id="block_blue")

    with pytest.raises(ValueError, match="pick target 'target' is not movable/graspable"):
        generator.build_ir_from_world("pick the target and place it on the green block surface", world)


def test_interactive_session_splits_semicolon_commands():
    commands = interactive_session.split_commands(
        "pick the red block and place it on the target surface; pick the blue block and place it on the red block surface ; ;"
    )
    assert commands == [
        "pick the red block and place it on the target surface",
        "pick the blue block and place it on the red block surface",
    ]


def test_interactive_session_deletes_default_latest_world_on_exit(monkeypatch):
    latest_path = world_sample_path("pytest_session_delete_latest")
    if latest_path.exists():
        latest_path.unlink()

    monkeypatch.setattr("builtins.input", lambda _prompt: "exit")

    result = interactive_session.main(["--out-prefix", "pytest_session_delete"])

    assert result == 0
    assert not latest_path.exists()


def test_interactive_session_preserves_explicit_saved_world(tmp_path: Path, monkeypatch):
    saved_world = tmp_path / "kept_latest.world.json"
    if saved_world.exists():
        saved_world.unlink()

    monkeypatch.setattr("builtins.input", lambda _prompt: "exit")

    result = interactive_session.main(["--out-prefix", "pytest_session_keep", "--save-final-world", str(saved_world)])

    assert result == 0
    assert saved_world.exists()


def test_interactive_session_deletes_generated_ir_on_exit(monkeypatch):
    ir_output = ir_sample_path("pytest_session_ir_delete-001")
    if ir_output.exists():
        ir_output.unlink()

    inputs = iter(["pick the red block and place it on the target surface", "exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    result = interactive_session.main(["--out-prefix", "pytest_session_ir_delete"])

    assert result == 0
    assert not ir_output.exists()


def test_interactive_session_save_session_keeps_generated_artifacts(monkeypatch):
    ir_output = ir_sample_path("pytest_session_keep_artifacts-001")
    latest_world = world_sample_path("pytest_session_keep_artifacts_latest")
    if ir_output.exists():
        ir_output.unlink()
    if latest_world.exists():
        latest_world.unlink()

    inputs = iter(["pick the red block and place it on the target surface", "exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    result = interactive_session.main(["--out-prefix", "pytest_session_keep_artifacts", "--save-session"])

    assert result == 0
    assert ir_output.exists()
    assert latest_world.exists()

    ir_output.unlink()
    latest_world.unlink()


def test_make_world_cli_builds_demo_world(tmp_path: Path):
    output_path = tmp_path / "demo_multi_blocks.world.json"

    result = make_world.main(
        [
            "--block",
            "red@0.30,0.10,0.02",
            "--block",
            "blue@0.30,-0.10,0.02",
            "--block",
            "green@0.45,0.10,0.02",
            "--fixture",
            "target_zone@0.55,0.12,0.01@0.20,0.15,0.02",
            "--out",
            str(output_path),
        ]
    )

    assert result == 0
    saved_world = WorldModel.model_validate(load_json(str(output_path)))
    assert set(saved_world.objects.keys()) == {"block_red", "block_blue", "block_green", "target_zone"}
    assert saved_world.features["target_surface"].parent_object == "target_zone"
    assert saved_world.features["block_red_top_surface"].local_pose.position == [0.0, 0.0, 0.02]
    assert saved_world.frames["block_blue_frame"].registry_path == "obj_block_blue_01/frame"
    assert saved_world.robot_state.tcp_pose.position == [0.4, 0.0, 0.3]


def test_make_world_cli_normalizes_target_alias(tmp_path: Path):
    output_path = tmp_path / "target_alias.world.json"

    result = make_world.main(
        [
            "--block",
            "red@0.30,0.10,0.02",
            "--fixture",
            "target@0.55,0.12,0.01",
            "--out",
            str(output_path),
        ]
    )

    assert result == 0
    saved_world = WorldModel.model_validate(load_json(str(output_path)))
    assert "target_zone" in saved_world.objects
    assert "target_surface" in saved_world.features


def test_make_world_cli_builds_from_yaml(tmp_path: Path):
    yaml_path = tmp_path / "demo_world.yaml"
    output_path = tmp_path / "demo_world.world.json"
    yaml_path.write_text(
        "\n".join(
            [
                "blocks:",
                "  - name: red",
                "    position: [0.30, 0.10, 0.02]",
                "  - name: blue",
                "    position: [0.30, -0.10, 0.02]",
                "fixtures:",
                "  - name: target",
                "    position: [0.55, 0.12, 0.01]",
                "    size: [0.20, 0.15, 0.02]",
                "tcp: [0.45, 0.0, 0.35]",
            ]
        ),
        encoding="utf-8",
    )

    result = make_world.main(["--yaml", str(yaml_path), "--out", str(output_path)])

    assert result == 0
    saved_world = WorldModel.model_validate(load_json(str(output_path)))
    assert set(saved_world.objects.keys()) == {"block_red", "block_blue", "target_zone"}
    assert saved_world.robot_state.tcp_pose.position == [0.45, 0.0, 0.35]
    assert saved_world.features["target_surface"].size_hint == [0.2, 0.15]


def test_make_world_cli_merges_yaml_and_cli_specs(tmp_path: Path):
    yaml_path = tmp_path / "base_world.yaml"
    output_path = tmp_path / "merged_world.world.json"
    yaml_path.write_text(
        "\n".join(
            [
                "blocks:",
                "  - name: red",
                "    position: [0.30, 0.10, 0.02]",
                "fixtures:",
                "  - name: target_zone",
                "    position: [0.55, 0.12, 0.01]",
            ]
        ),
        encoding="utf-8",
    )

    result = make_world.main(
        [
            "--yaml",
            str(yaml_path),
            "--block",
            "green@0.45,0.10,0.02",
            "--out",
            str(output_path),
        ]
    )

    assert result == 0
    saved_world = WorldModel.model_validate(load_json(str(output_path)))
    assert set(saved_world.objects.keys()) == {"block_red", "block_green", "target_zone"}


def test_existing_world_reuse_skips_derived_world_file(tmp_path: Path):
    prefix = "pytest_existing_world_reuse"
    ir_output = ir_sample_path(prefix)
    world_output = world_sample_path(prefix)
    if ir_output.exists():
        ir_output.unlink()
    if world_output.exists():
        world_output.unlink()

    existing_world = tmp_path / "existing.world.json"
    existing_world.write_text(
        json.dumps(load_json(str(world_sample_path("sample_pick_place"))), indent=2),
        encoding="utf-8",
    )

    result = generator.main(
        [
            "--nl",
            "pick red block at (0.3,0.1,0.02) and place it at (0.55,0.12,0.01)",
            "--existing-world",
            str(existing_world),
            "--out-prefix",
            prefix,
            "--sim-backend",
            "mock",
            "--force",
        ]
    )

    assert result == 0
    assert ir_output.exists()
    assert not world_output.exists()


def test_existing_world_no_run_persists_derived_world_when_adjusted(tmp_path: Path):
    prefix = "pytest_existing_world_adjusted"
    ir_output = ir_sample_path(prefix)
    world_output = world_sample_path(prefix)
    if ir_output.exists():
        ir_output.unlink()
    if world_output.exists():
        world_output.unlink()

    existing_world = tmp_path / "existing_adjusted.world.json"
    existing_world.write_text(
        json.dumps(load_json(str(world_sample_path("demo_multi_blocks"))), indent=2),
        encoding="utf-8",
    )

    result = generator.main(
        [
            "--nl",
            "pick the blue block and place it on the target surface",
            "--existing-world",
            str(existing_world),
            "--out-prefix",
            prefix,
            "--no-run",
            "--force",
        ]
    )

    assert result == 0
    assert ir_output.exists()
    assert world_output.exists()


def test_run_ir_with_final_world_supports_session_chaining():
    world = WorldModel.model_validate(load_json(str(world_sample_path("demo_multi_blocks"))))
    prepared_world = generator.ensure_world_features_and_frames(world, source_object_id="block_red")
    red_ir = GenericCobotIR.model_validate(
        generator.build_ir_from_world("pick the red block and place it on the target surface", prepared_world)
    )

    result, final_world = run_ir_with_final_world(red_ir, world_model=prepared_world, simulator_backend="mock")

    assert result.status == "passed"
    assert final_world is not None
    assert final_world.objects["block_red"].pose.position[:2] == [0.55, 0.12]
    assert final_world.robot_state.tcp_pose.position == [0.4, 0.0, 0.3]

    chained_world = generator.ensure_world_features_and_frames(final_world, source_object_id="block_blue")
    blue_ir = GenericCobotIR.model_validate(
        generator.build_ir_from_world("pick the blue block and place it on the target surface", chained_world)
    )
    assert blue_ir.action_plan[0].inputs["object"] == "block_blue"


def test_run_ir_with_final_world_supports_stacking_sequence():
    world = WorldModel.model_validate(load_json(str(world_sample_path("demo_multi_blocks"))))

    red_world = generator.ensure_world_features_and_frames(world, source_object_id="block_red")
    red_ir = GenericCobotIR.model_validate(
        generator.build_ir_from_world("pick the red block and place it on the target surface", red_world)
    )
    red_result, world_after_red = run_ir_with_final_world(red_ir, world_model=red_world, simulator_backend="mock")

    assert red_result.status == "passed"
    assert world_after_red is not None

    blue_world = generator.ensure_world_features_and_frames(
        world_after_red,
        source_object_id="block_blue",
        target_object_id="block_red",
    )
    blue_ir = GenericCobotIR.model_validate(
        generator.build_ir_from_world("pick the blue block and place it on the red block surface", blue_world)
    )
    blue_result, world_after_blue = run_ir_with_final_world(blue_ir, world_model=blue_world, simulator_backend="mock")

    assert blue_result.status == "passed"
    assert world_after_blue is not None
    assert world_after_blue.objects["block_blue"].pose.position == [0.55, 0.12, 0.08]

    green_world = generator.ensure_world_features_and_frames(
        world_after_blue,
        source_object_id="block_green",
        target_object_id="block_blue",
    )
    green_ir = GenericCobotIR.model_validate(
        generator.build_ir_from_world("pick the green block and place it on the blue block surface", green_world)
    )
    green_result, world_after_green = run_ir_with_final_world(green_ir, world_model=green_world, simulator_backend="mock")

    assert green_result.status == "passed"
    assert world_after_green is not None
    assert world_after_green.objects["block_green"].pose.position == pytest.approx([0.55, 0.12, 0.12])

    pickup_world = generator.ensure_world_features_and_frames(world_after_green, source_object_id="block_green")
    pickup_ir = GenericCobotIR.model_validate(
        generator.build_ir_from_world("pick the green block and place it on the target surface", pickup_world)
    )
    pickup_result, world_after_pickup = run_ir_with_final_world(pickup_ir, world_model=pickup_world, simulator_backend="mock")

    assert pickup_result.status == "passed"
    assert world_after_pickup is not None
    assert pickup_world.get_object("block_green").pose.position == pytest.approx([0.55, 0.12, 0.12])
    assert pickup_ir.action_plan[2].inputs["approach_pose"]["offset"][2] > 0.04
