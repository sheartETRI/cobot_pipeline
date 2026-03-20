import pytest

from geometry_utils import (
    aabb_intersect,
    compute_aabb_from_box,
    compose_pose,
    quat_mul,
    quat_rotate_vec,
)


def test_quaternion_rotation_identity():
    rotated = quat_rotate_vec([0.0, 0.0, 0.0, 1.0], [1.0, 2.0, 3.0])
    assert rotated == pytest.approx([1.0, 2.0, 3.0])


def test_compose_pose_translation_and_rotation():
    parent = {
        "frame": "world",
        "position": [1.0, 0.0, 0.0],
        "orientation": [0.0, 0.0, 0.70710678, 0.70710678],
    }
    child = {
        "frame": "frame_a",
        "position": [1.0, 0.0, 0.0],
        "orientation": [0.0, 0.0, 0.0, 1.0],
    }
    world_pose = compose_pose(parent, child)
    assert world_pose["position"] == pytest.approx([1.0, 1.0, 0.0], abs=1e-6)


def test_compute_aabb_box_no_rotation():
    minv, maxv = compute_aabb_from_box(
        center_world=[0.5, 0.0, 0.1],
        orientation=[0.0, 0.0, 0.0, 1.0],
        size=[0.2, 0.1, 0.05],
    )
    assert minv == pytest.approx([0.4, -0.05, 0.075])
    assert maxv == pytest.approx([0.6, 0.05, 0.125])


def test_compute_aabb_box_with_rotation():
    minv, maxv = compute_aabb_from_box(
        center_world=[0.0, 0.0, 0.0],
        orientation=[0.0, 0.0, 0.70710678, 0.70710678],
        size=[0.2, 0.1, 0.05],
    )
    assert minv[0] == pytest.approx(-0.05, abs=1e-5)
    assert maxv[1] == pytest.approx(0.1, abs=1e-5)


def test_aabb_intersection():
    assert aabb_intersect([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [1.5, 1.5, 1.5])
    assert not aabb_intersect([0.0, 0.0, 0.0], [0.4, 0.4, 0.4], [0.5, 0.5, 0.5], [1.5, 1.5, 1.5])
