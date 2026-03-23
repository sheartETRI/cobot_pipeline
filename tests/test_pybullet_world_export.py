import json
from pathlib import Path

import pytest

from ir_models import GenericCobotIR
from run_demo import validate_ir_world_consistency
from sample_paths import exported_world_sample_path, ir_sample_path, world_sample_path
from scripts import generate_and_run_rule as generator
from scripts import pybullet_world_utils
from world_model import WorldModel


pybullet = pytest.importorskip("pybullet")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def cleanup_pybullet():
    pybullet_world_utils.disconnect()
    yield
    pybullet_world_utils.disconnect()


def test_export_world_model_validates(tmp_path: Path):
    pybullet_world_utils.connect(gui=False)
    pybullet_world_utils.create_box("obj_block_red_01", (0.3, 0.1, 0.02))
    pybullet_world_utils.create_fixture_box("obj_target_zone_01", (0.55, 0.12, 0.01))
    pybullet_world_utils.add_feature("block_red_top_surface", "obj_block_red_01", (0.0, 0.0, 0.02), size_hint=[0.04, 0.04])
    pybullet_world_utils.add_feature("target_surface", "obj_target_zone_01", (0.0, 0.0, 0.01), size_hint=[0.2, 0.15])

    exported = pybullet_world_utils.export_world_model(str(tmp_path / "exported.world.json"))
    world = WorldModel.model_validate(load_json(Path(exported)))

    assert "block_red" in world.objects
    assert "target_zone" in world.objects
    assert "block_red_top_surface" in world.features
    assert "target_surface" in world.features
    assert "block_red_frame" in world.frames
    assert "target_zone_frame" in world.frames


def test_generate_and_run_with_world_from_pybullet(tmp_path: Path):
    pybullet_world_utils.connect(gui=False)
    pybullet_world_utils.create_box("obj_block_red_01", (0.3, 0.1, 0.02))
    pybullet_world_utils.create_fixture_box("obj_target_zone_01", (0.55, 0.12, 0.01))
    pybullet_world_utils.add_feature("block_red_top_surface", "obj_block_red_01", (0.0, 0.0, 0.02), size_hint=[0.04, 0.04])
    pybullet_world_utils.add_feature("target_surface", "obj_target_zone_01", (0.0, 0.0, 0.01), size_hint=[0.2, 0.15])

    prefix = "pytest_pybullet_world"
    result = generator.main(
        [
            "--nl",
            "pick red block at (0.3,0.1,0.02) and place it at (0.55,0.12,0.01)",
            "--world-from-pybullet",
            "--out-prefix",
            prefix,
            "--sim-backend",
            "mock",
            "--force",
        ]
    )

    assert result == 0

    ir_path = ir_sample_path(prefix)
    world_path = world_sample_path(prefix)
    exported_path = exported_world_sample_path(prefix)

    assert ir_path.exists()
    assert world_path.exists()
    assert exported_path.exists()

    ir = GenericCobotIR.model_validate(load_json(ir_path))
    world = WorldModel.model_validate(load_json(world_path))
    errors, warnings = validate_ir_world_consistency(ir, world)

    assert errors == []
    assert warnings == []
