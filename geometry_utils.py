from __future__ import annotations

from typing import Dict, List, Tuple


Vector3 = List[float]
Quaternion = List[float]
PoseDict = Dict[str, object]
AABB = Tuple[Vector3, Vector3]


def quat_mul(a: Quaternion, b: Quaternion) -> Quaternion:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def quat_norm(q: Quaternion) -> float:
    return sum(component * component for component in q) ** 0.5


def quat_normalize(q: Quaternion) -> Quaternion:
    norm = quat_norm(q)
    if norm == 0.0:
        return [0.0, 0.0, 0.0, 1.0]
    return [component / norm for component in q]


def quat_rotate_vec(q: Quaternion, v: Vector3) -> Vector3:
    qx, qy, qz, qw = quat_normalize(q)
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    xw = qx * qw
    yw = qy * qw
    zw = qz * qw
    r00 = 1 - 2 * (yy + zz)
    r01 = 2 * (xy - zw)
    r02 = 2 * (xz + yw)
    r10 = 2 * (xy + zw)
    r11 = 1 - 2 * (xx + zz)
    r12 = 2 * (yz - xw)
    r20 = 2 * (xz - yw)
    r21 = 2 * (yz + xw)
    r22 = 1 - 2 * (xx + yy)
    x = r00 * v[0] + r01 * v[1] + r02 * v[2]
    y = r10 * v[0] + r11 * v[1] + r12 * v[2]
    z = r20 * v[0] + r21 * v[1] + r22 * v[2]
    return [x, y, z]


def compose_pose(p_parent: PoseDict, p_child: PoseDict) -> PoseDict:
    parent_position = p_parent["position"]
    parent_orientation = p_parent["orientation"]
    child_position = p_child["position"]
    child_orientation = p_child["orientation"]
    rotated_child = quat_rotate_vec(parent_orientation, child_position)
    world_position = [
        parent_position[0] + rotated_child[0],
        parent_position[1] + rotated_child[1],
        parent_position[2] + rotated_child[2],
    ]
    world_orientation = quat_mul(parent_orientation, child_orientation)
    return {
        "frame": p_parent.get("frame", "world"),
        "position": world_position,
        "orientation": quat_normalize(world_orientation),
    }


def identity_pose(frame: str = "world") -> PoseDict:
    return {
        "frame": frame,
        "position": [0.0, 0.0, 0.0],
        "orientation": [0.0, 0.0, 0.0, 1.0],
    }


def get_box_corners(size: Vector3) -> List[Vector3]:
    sx, sy, sz = size
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    return [
        [dx, dy, dz]
        for dx in (-hx, hx)
        for dy in (-hy, hy)
        for dz in (-hz, hz)
    ]


def compute_aabb_from_box(center_world: Vector3, orientation: Quaternion, size: Vector3) -> AABB:
    corners_world: List[Vector3] = []
    for corner in get_box_corners(size):
        rotated_corner = quat_rotate_vec(orientation, corner)
        corners_world.append(
            [
                center_world[0] + rotated_corner[0],
                center_world[1] + rotated_corner[1],
                center_world[2] + rotated_corner[2],
            ]
        )
    min_xyz = [min(point[i] for point in corners_world) for i in range(3)]
    max_xyz = [max(point[i] for point in corners_world) for i in range(3)]
    return min_xyz, max_xyz


def aabb_intersect(a_min: Vector3, a_max: Vector3, b_min: Vector3, b_max: Vector3) -> bool:
    for i in range(3):
        if a_max[i] < b_min[i] or a_min[i] > b_max[i]:
            return False
    return True
