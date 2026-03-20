from __future__ import annotations

import logging
from importlib.util import find_spec
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from geometry_utils import aabb_intersect, compute_aabb_from_box, compose_pose, identity_pose
from world_model import BoxGeometry, CylinderGeometry, FeatureModel, ObjectModel, PlaneGeometry, Pose, WorldModel

from ir_models import (
    ActionStep,
    ErrorSeverity,
    GenericCobotIR,
    PrimitiveType,
    StepStatus,
    StepTrace,
    VerificationError,
    VerificationMetrics,
    VerificationResult,
    VerificationSummary,
)


logger = logging.getLogger(__name__)

WorldPose = Dict[str, object]
AABB = Tuple[List[float], List[float]]
DEFAULT_GRIPPER_SIZE = [0.08, 0.08, 0.12]


@dataclass
class MockWorldState:
    bound_objects: set[str] = field(default_factory=set)
    known_features: set[str] = field(default_factory=set)
    attached_object: str | None = None
    achieved_effects: set[str] = field(default_factory=set)
    object_world_poses: Dict[str, WorldPose] = field(default_factory=dict)
    object_aabbs: Dict[str, AABB] = field(default_factory=dict)
    tcp_pose_world: Optional[WorldPose] = None


def pose_model_to_dict(pose: Pose) -> WorldPose:
    return {
        "frame": pose.frame,
        "position": list(pose.position),
        "orientation": list(pose.orientation),
    }


class MockSimulator:
    """
    Minimal mock simulator with world-model-aware reference validation and
    approximate AABB collision checks.
    """

    def __init__(self, ir: GenericCobotIR, world_model: WorldModel | None = None) -> None:
        self.ir = ir
        self.world_model = world_model
        self.gripper_size = DEFAULT_GRIPPER_SIZE[:]

        known_features = set(ir.world_binding.features.keys())
        bound_objects: set[str] = set()
        attached_object: Optional[str] = None
        object_world_poses: Dict[str, WorldPose] = {}
        object_aabbs: Dict[str, AABB] = {}
        tcp_pose_world: Optional[WorldPose] = None

        if world_model is not None:
            known_features.update(world_model.features.keys())
            bound_objects.update(world_model.objects.keys())

            for alias, registry_id in ir.world_binding.objects.items():
                matched = world_model.find_object_by_registry_id(registry_id)
                if matched is not None and matched.object_id == alias:
                    bound_objects.add(alias)
                    logger.debug("Bound world object by registry_id: alias=%s registry_id=%s", alias, registry_id)

            for object_id in world_model.objects:
                world_pose = self.resolve_pose_to_world(world_model.get_object(object_id).pose)
                object_world_poses[object_id] = world_pose
                object_aabbs[object_id] = self.compute_object_aabb(object_id, world_pose)

            tcp_pose_world = self.resolve_pose_to_world(world_model.robot_state.tcp_pose)
            attached_object = world_model.robot_state.attached_object

        self.world = MockWorldState(
            bound_objects=bound_objects,
            known_features=known_features,
            attached_object=attached_object,
            object_world_poses=object_world_poses,
            object_aabbs=object_aabbs,
            tcp_pose_world=tcp_pose_world,
        )
        self.simulator_name = "mock_sim"
        logger.debug(
            "Initialized MockSimulator bound_objects=%s known_features=%s attached=%s tcp_pose=%s",
            sorted(bound_objects),
            sorted(known_features),
            attached_object,
            tcp_pose_world,
        )

    def geometry_size(self, obj: ObjectModel) -> List[float]:
        geometry = obj.geometry
        if isinstance(geometry, BoxGeometry):
            return list(geometry.size)
        if isinstance(geometry, CylinderGeometry):
            return [geometry.radius * 2.0, geometry.radius * 2.0, geometry.height]
        if isinstance(geometry, PlaneGeometry):
            return [geometry.size[0], geometry.size[1], 0.002]
        return [0.02, 0.02, 0.02]

    def resolve_pose_to_world(self, pose: Pose | WorldPose, visited: Optional[set[str]] = None) -> WorldPose:
        if isinstance(pose, Pose):
            pose_dict = pose_model_to_dict(pose)
        else:
            pose_dict = {
                "frame": pose["frame"],
                "position": list(pose["position"]),
                "orientation": list(pose["orientation"]),
            }

        if self.world_model is None:
            return pose_dict

        frame_name = str(pose_dict["frame"])
        if frame_name in {self.world_model.world_frame, "world"}:
            return {
                "frame": self.world_model.world_frame,
                "position": list(pose_dict["position"]),
                "orientation": list(pose_dict["orientation"]),
            }

        if visited is None:
            visited = set()
        if frame_name in visited:
            raise ValueError(f"cyclic frame reference detected: {frame_name}")
        next_visited = set(visited)
        next_visited.add(frame_name)

        parent_pose: WorldPose | None = None
        if self.world_model.has_frame(frame_name):
            parent_pose = pose_model_to_dict(self.world_model.get_frame(frame_name).pose)
        elif self.world_model.has_object(frame_name):
            parent_pose = pose_model_to_dict(self.world_model.get_object(frame_name).pose)
        elif self.world_model.has_feature(frame_name):
            parent_pose = pose_model_to_dict(self.world_model.get_feature(frame_name).local_pose)
        else:
            for frame in self.world_model.frames.values():
                if frame.registry_path == frame_name:
                    parent_pose = pose_model_to_dict(frame.pose)
                    break

        if parent_pose is None:
            raise KeyError(f"unknown frame reference: {frame_name}")

        parent_world_pose = self.resolve_pose_to_world(parent_pose, visited=next_visited)
        local_pose = {
            "frame": parent_world_pose["frame"],
            "position": list(pose_dict["position"]),
            "orientation": list(pose_dict["orientation"]),
        }
        return compose_pose(parent_world_pose, local_pose)

    def resolve_pose_ref_to_world(self, pose_ref: Any) -> Optional[WorldPose]:
        if self.world_model is None:
            return None

        ref_name = pose_ref.ref
        if ref_name in {self.world_model.world_frame, "world"}:
            base_pose = identity_pose(self.world_model.world_frame)
        elif self.world_model.has_frame(ref_name):
            base_pose = self.resolve_pose_to_world(self.world_model.get_frame(ref_name).pose)
        elif self.world_model.has_object(ref_name):
            base_pose = self.resolve_pose_to_world(self.world_model.get_object(ref_name).pose)
        elif self.world_model.has_feature(ref_name):
            base_pose = self.resolve_pose_to_world(self.world_model.get_feature(ref_name).local_pose)
        else:
            matched_frame = next(
                (frame for frame in self.world_model.frames.values() if frame.registry_path == ref_name),
                None,
            )
            if matched_frame is None:
                return None
            base_pose = self.resolve_pose_to_world(matched_frame.pose)

        offset_pose = {
            "frame": base_pose["frame"],
            "position": list(getattr(pose_ref, "offset", [0.0, 0.0, 0.0])),
            "orientation": [0.0, 0.0, 0.0, 1.0],
        }
        return compose_pose(base_pose, offset_pose)

    def resolve_feature_pose_to_world(self, feature_id: str) -> Optional[WorldPose]:
        if self.world_model is None or not self.world_model.has_feature(feature_id):
            return None
        feature: FeatureModel = self.world_model.get_feature(feature_id)
        return self.resolve_pose_to_world(feature.local_pose)

    def compute_object_aabb(self, object_id: str, world_pose: Optional[WorldPose] = None) -> AABB:
        if self.world_model is None:
            raise KeyError("world_model is required to compute object AABB")
        obj = self.world_model.get_object(object_id)
        if world_pose is None:
            world_pose = self.resolve_pose_to_world(obj.pose)
        return compute_aabb_from_box(
            center_world=list(world_pose["position"]),
            orientation=list(world_pose["orientation"]),
            size=self.geometry_size(obj),
        )

    def compute_gripper_aabb(self, world_pose: WorldPose) -> AABB:
        return compute_aabb_from_box(
            center_world=list(world_pose["position"]),
            orientation=list(world_pose["orientation"]),
            size=self.gripper_size,
        )

    def update_object_world_pose(self, object_id: str, world_pose: WorldPose) -> None:
        self.world.object_world_poses[object_id] = world_pose
        self.world.object_aabbs[object_id] = self.compute_object_aabb(object_id, world_pose)
        logger.debug("Updated world pose for object=%s pose=%s aabb=%s", object_id, world_pose, self.world.object_aabbs[object_id])

    def infer_owner_for_ref(self, ref_name: str) -> Optional[str]:
        if self.world_model is None:
            return None
        if self.world_model.has_object(ref_name):
            return ref_name
        if self.world_model.has_feature(ref_name):
            return self.world_model.get_feature(ref_name).parent_object
        if self.world_model.has_frame(ref_name):
            registry_base = self.world_model.get_frame(ref_name).registry_path.split("/")[0]
            matched = self.world_model.find_object_by_registry_id(registry_base)
            return matched.object_id if matched is not None else None
        for frame in self.world_model.frames.values():
            if frame.registry_path == ref_name:
                registry_base = frame.registry_path.split("/")[0]
                matched = self.world_model.find_object_by_registry_id(registry_base)
                return matched.object_id if matched is not None else None
        return None

    def build_collision_error(self, step_type: PrimitiveType, entities: List[str], collided_with: str) -> Dict[str, Any]:
        return {
            "error_id": "e_aabb_collision",
            "type": "collision",
            "message": f"AABB collision detected during {step_type.value} with object '{collided_with}'",
            "entities": entities + [collided_with],
            "suggested_repairs": [
                "increase_clearance",
                "replan_path",
                "adjust_target_pose",
            ],
        }

    def check_aabb_collision(
        self,
        moving_aabb: AABB,
        excluded_objects: set[str],
        step_type: PrimitiveType,
        moving_entities: List[str],
    ) -> Dict[str, Any] | None:
        if self.world_model is None:
            return None
        moving_min, moving_max = moving_aabb
        logger.debug(
            "Checking AABB collision step=%s moving_aabb=%s excluded=%s",
            step_type.value,
            moving_aabb,
            sorted(excluded_objects),
        )
        for object_id, obj in self.world_model.objects.items():
            if object_id in excluded_objects or not obj.collision_enabled:
                continue
            obstacle_aabb = self.world.object_aabbs.get(object_id)
            if obstacle_aabb is None:
                continue
            if aabb_intersect(moving_min, moving_max, obstacle_aabb[0], obstacle_aabb[1]):
                logger.debug(
                    "AABB collision step=%s moving=%s obstacle=%s obstacle_aabb=%s",
                    step_type.value,
                    moving_entities,
                    object_id,
                    obstacle_aabb,
                )
                return self.build_collision_error(step_type, moving_entities, object_id)
        return None

    def check_pose_collision(
        self,
        step_type: PrimitiveType,
        target_pose_world: Optional[WorldPose],
        excluded_objects: set[str],
        attached_object: Optional[str] = None,
    ) -> Dict[str, Any] | None:
        if self.world_model is None or target_pose_world is None:
            return None

        gripper_aabb = self.compute_gripper_aabb(target_pose_world)
        collision = self.check_aabb_collision(
            moving_aabb=gripper_aabb,
            excluded_objects=excluded_objects,
            step_type=step_type,
            moving_entities=["gripper_tcp"],
        )
        if collision is not None:
            return collision

        if attached_object is not None and self.world_model.has_object(attached_object):
            attached_aabb = self.compute_object_aabb(attached_object, target_pose_world)
            collision = self.check_aabb_collision(
                moving_aabb=attached_aabb,
                excluded_objects=excluded_objects | {attached_object},
                step_type=step_type,
                moving_entities=[attached_object],
            )
            if collision is not None:
                return collision
        return None

    def check_insert_geometry_collision(self, step_inputs: Any) -> Dict[str, Any] | None:
        if self.world_model is None:
            return None
        if not self.world_model.has_object(step_inputs.source_object):
            return None
        if not step_inputs.target_feature or not self.world_model.has_feature(step_inputs.target_feature):
            return None

        source_object = self.world_model.get_object(step_inputs.source_object)
        target_feature = self.world_model.get_feature(step_inputs.target_feature)
        source_size = self.geometry_size(source_object)
        source_diameter = max(source_size[0], source_size[1])
        feature_width = target_feature.width
        feature_depth = target_feature.depth
        logger.debug(
            "Checking insert geometry source=%s size=%s target_feature=%s width=%s depth=%s insert_depth=%.4f",
            step_inputs.source_object,
            source_size,
            step_inputs.target_feature,
            feature_width,
            feature_depth,
            step_inputs.insert_depth,
        )

        if feature_width is not None and source_diameter > feature_width:
            return {
                "error_id": "e_insert_geometry_collision",
                "type": "collision",
                "message": "source object cross-section is larger than target feature width",
                "entities": [step_inputs.source_object, step_inputs.target_feature],
                "suggested_repairs": [
                    "choose_smaller_object",
                    "choose_larger_target_feature",
                    "refine_world_geometry",
                ],
            }

        if feature_depth is not None and step_inputs.insert_depth > feature_depth:
            return {
                "error_id": "e_insert_geometry_collision",
                "type": "collision",
                "message": "insert depth exceeds target feature depth",
                "entities": [step_inputs.source_object, step_inputs.target_feature],
                "suggested_repairs": [
                    "reduce_insert_depth",
                    "choose_deeper_target_feature",
                    "refine_world_geometry",
                ],
            }

        return None

    def snapshot_state(self) -> Dict[str, Any]:
        return {
            "bound_objects": sorted(self.world.bound_objects),
            "known_features": sorted(self.world.known_features),
            "attached_object": self.world.attached_object,
            "achieved_effects": sorted(self.world.achieved_effects),
            "tcp_pose_world": self.world.tcp_pose_world,
        }

    def execute_step(self, step: ActionStep) -> Tuple[bool, Dict[str, Any] | None]:
        parsed_inputs = step.parsed_inputs()
        parsed_constraints = step.parsed_constraints()
        step_type = step.type

        logger.debug(
            "Executing step step_id=%s type=%s inputs=%s constraints=%s",
            step.step_id,
            step_type.value,
            step.inputs,
            step.constraints,
        )

        if step_type == PrimitiveType.FIND_OBJECT:
            obj_name = parsed_inputs.object
            self.world.bound_objects.add(obj_name)
            self.world.achieved_effects.add(f"object_pose_bound:{obj_name}")
            logger.debug("find_object succeeded for %s", obj_name)
            return True, None

        if step_type == PrimitiveType.APPROACH:
            target = parsed_inputs.target_object
            if target not in self.world.bound_objects:
                return False, {
                    "error_id": "e_approach_missing_binding",
                    "type": "binding_error",
                    "message": f"target object '{target}' is not bound",
                    "entities": [target],
                    "suggested_repairs": ["add_find_object_step"],
                }
            if parsed_inputs.target_feature and parsed_inputs.target_feature not in self.world.known_features:
                return False, {
                    "error_id": "e_approach_unknown_feature",
                    "type": "binding_error",
                    "message": f"target feature '{parsed_inputs.target_feature}' is not declared",
                    "entities": [parsed_inputs.target_feature],
                    "suggested_repairs": ["declare_feature_in_world_binding"],
                }

            approach_pose_world = self.resolve_pose_ref_to_world(parsed_inputs.approach_pose)
            excluded = {target}
            owner = self.infer_owner_for_ref(parsed_inputs.approach_pose.ref)
            if owner is not None:
                excluded.add(owner)
            collision = self.check_pose_collision(
                step_type=step_type,
                target_pose_world=approach_pose_world,
                excluded_objects=excluded,
                attached_object=self.world.attached_object,
            )
            if collision is not None:
                return False, collision

            self.world.tcp_pose_world = approach_pose_world
            if parsed_inputs.target_feature:
                self.world.achieved_effects.add(f"tcp_at_pregrasp_pose:{parsed_inputs.target_feature}")
            else:
                self.world.achieved_effects.add("tcp_at_pregrasp_pose")
            logger.debug("approach succeeded for target=%s pose=%s", target, approach_pose_world)
            return True, None

        if step_type == PrimitiveType.GRASP:
            target = parsed_inputs.target_object
            if target not in self.world.bound_objects:
                return False, {
                    "error_id": "e_grasp_missing_binding",
                    "type": "binding_error",
                    "message": f"cannot grasp unbound object '{target}'",
                    "entities": [target],
                    "suggested_repairs": ["add_find_object_step"],
                }
            self.world.attached_object = target
            if self.world.tcp_pose_world is not None:
                self.update_object_world_pose(target, self.world.tcp_pose_world)
            self.world.achieved_effects.add(f"object_attached:{target}")
            if parsed_inputs.target_feature:
                self.world.achieved_effects.add(f"grasp_target_feature:{target}:{parsed_inputs.target_feature}")
            logger.debug("grasp succeeded for target=%s", target)
            return True, None

        if step_type == PrimitiveType.RETREAT:
            return True, None

        if step_type == PrimitiveType.MOVE_LINEAR:
            target_pose_world = self.resolve_pose_ref_to_world(parsed_inputs.target_pose)
            owner = self.infer_owner_for_ref(parsed_inputs.target_pose.ref)
            excluded = set()
            if self.world.attached_object is not None:
                excluded.add(self.world.attached_object)
            if owner is not None:
                excluded.add(owner)
            collision = self.check_pose_collision(
                step_type=step_type,
                target_pose_world=target_pose_world,
                excluded_objects=excluded,
                attached_object=self.world.attached_object,
            )
            if collision is not None:
                return False, collision

            self.world.tcp_pose_world = target_pose_world
            if self.world.attached_object is not None and target_pose_world is not None:
                self.update_object_world_pose(self.world.attached_object, target_pose_world)
            logger.debug("move_linear succeeded to pose=%s", target_pose_world)
            return True, None

        if step_type == PrimitiveType.PLACE:
            target = parsed_inputs.target_object
            if self.world.attached_object != target:
                return False, {
                    "error_id": "e_place_not_attached",
                    "type": "precondition_error",
                    "message": f"object '{target}' is not currently attached",
                    "entities": [target],
                    "suggested_repairs": ["add_grasp_step", "check_grasp_success"],
                }

            destination_pose_world = self.resolve_pose_ref_to_world(parsed_inputs.destination_pose)
            owner = self.infer_owner_for_ref(parsed_inputs.destination_pose.ref)
            excluded = {target}
            if owner is not None:
                excluded.add(owner)
            collision = self.check_pose_collision(
                step_type=step_type,
                target_pose_world=destination_pose_world,
                excluded_objects=excluded,
                attached_object=target,
            )
            if collision is not None:
                return False, collision

            if destination_pose_world is not None:
                self.world.tcp_pose_world = destination_pose_world
                self.update_object_world_pose(target, destination_pose_world)
            effect_suffix = (
                f"{target}:{parsed_inputs.target_feature}"
                if parsed_inputs.target_feature
                else target
            )
            self.world.achieved_effects.add(f"object_at_destination:{effect_suffix}")
            logger.debug("place succeeded for target=%s destination=%s", target, destination_pose_world)
            return True, None

        if step_type == PrimitiveType.ALIGN:
            target_suffix = (
                parsed_inputs.target_feature
                if parsed_inputs.target_feature
                else parsed_inputs.target_object
            )
            source_suffix = (
                parsed_inputs.source_feature
                if parsed_inputs.source_feature
                else parsed_inputs.source_object
            )
            self.world.achieved_effects.add(f"objects_aligned:{source_suffix}:{target_suffix}")
            logger.debug("align succeeded source=%s target=%s", source_suffix, target_suffix)
            return True, None

        if step_type == PrimitiveType.INSERT:
            geometry_collision = self.check_insert_geometry_collision(parsed_inputs)
            if geometry_collision is not None:
                return False, geometry_collision

            target_feature_pose_world = (
                self.resolve_feature_pose_to_world(parsed_inputs.target_feature)
                if parsed_inputs.target_feature
                else None
            )
            excluded = {parsed_inputs.source_object, parsed_inputs.target_object}
            collision = self.check_pose_collision(
                step_type=step_type,
                target_pose_world=target_feature_pose_world,
                excluded_objects=excluded,
                attached_object=parsed_inputs.source_object,
            )
            if collision is not None:
                return False, collision

            speed = getattr(parsed_constraints, "speed", 0.03)
            max_force = getattr(parsed_constraints, "max_force", None)
            if speed > 0.03 or (max_force is not None and max_force > 15):
                collision_entity = (
                    parsed_inputs.target_feature
                    if parsed_inputs.target_feature
                    else parsed_inputs.target_object
                )
                return False, {
                    "error_id": "e_insert_collision",
                    "type": "collision",
                    "message": "collision during insertion due to aggressive insertion parameters",
                    "entities": ["gripper_tcp", collision_entity],
                    "suggested_repairs": [
                        "increase_preinsert_height",
                        "reduce_insert_speed",
                        "tighten_alignment_tolerance",
                    ],
                }

            if target_feature_pose_world is not None:
                self.world.tcp_pose_world = target_feature_pose_world
                self.update_object_world_pose(parsed_inputs.source_object, target_feature_pose_world)

            target_suffix = (
                parsed_inputs.target_feature
                if parsed_inputs.target_feature
                else parsed_inputs.target_object
            )
            self.world.achieved_effects.add(f"{parsed_inputs.source_object}_inserted_into_{target_suffix}")
            logger.debug("insert succeeded source=%s target=%s", parsed_inputs.source_object, target_suffix)
            return True, None

        if step_type == PrimitiveType.RELEASE:
            target = parsed_inputs.target_object
            if self.world.attached_object == target:
                self.world.attached_object = None
            self.world.achieved_effects.add("release_completed")
            self.world.achieved_effects.add(f"object_detached:{target}")
            logger.debug("release succeeded target=%s", target)
            return True, None

        if step_type == PrimitiveType.WAIT:
            return True, None

        if step_type == PrimitiveType.CHECK:
            condition = parsed_inputs.condition
            must_be_true = getattr(parsed_constraints, "must_be_true", True)
            exists = condition in self.world.achieved_effects
            if must_be_true and not exists:
                return False, {
                    "error_id": "e_check_failed",
                    "type": "check_failed",
                    "message": f"required condition not satisfied: {condition}",
                    "entities": [condition],
                    "suggested_repairs": ["replan_previous_step"],
                }
            logger.debug("check succeeded condition=%s", condition)
            return True, None

        if step_type == PrimitiveType.MOVE_JOINT:
            return True, None

        return False, {
            "error_id": "e_unknown_step",
            "type": "unsupported_step",
            "message": f"unsupported step type: {step_type}",
            "entities": [str(step_type)],
            "suggested_repairs": ["implement_executor_handler"],
        }


class PyBulletSimulator(MockSimulator):
    """
    Demo-level PyBullet backend.
    Instead of a full robot model, it simulates a floating TCP proxy body,
    world objects, and grasp constraints. This is enough to demonstrate
    primitive-to-simulator mapping and real collision queries.
    """

    def __init__(
        self,
        ir: GenericCobotIR,
        world_model: WorldModel | None = None,
        *,
        gui: bool = False,
        path_samples: int = 12,
    ) -> None:
        if world_model is None:
            raise ValueError("PyBulletSimulator requires a world_model")
        if find_spec("pybullet") is None:
            raise RuntimeError(
                "pybullet is not installed. Install it with `pip install pybullet` "
                "or run with simulator_backend='mock'."
            )

        super().__init__(ir, world_model=world_model)

        import pybullet as pybullet

        self.p = pybullet
        self.client_id = self.p.connect(self.p.GUI if gui else self.p.DIRECT)
        self.path_samples = max(path_samples, 2)
        self.simulator_name = "pybullet_sim"
        self.body_ids: Dict[str, int] = {}
        self.tcp_body_id: Optional[int] = None
        self.grasp_constraint_id: Optional[int] = None
        self.gripper_half_height = self.gripper_size[2] / 2.0

        self.p.resetSimulation(physicsClientId=self.client_id)
        self.p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)
        self.p.setTimeStep(1.0 / 240.0, physicsClientId=self.client_id)

        self._build_world_bodies()
        self._build_tcp_proxy()

    def close(self) -> None:
        if getattr(self, "client_id", None) is not None:
            try:
                self.p.disconnect(physicsClientId=self.client_id)
            except Exception:
                logger.debug("Ignoring PyBullet disconnect failure", exc_info=True)
            self.client_id = None

    def geometry_shape(self, geometry: Any) -> tuple[int, list[float], float]:
        if isinstance(geometry, BoxGeometry):
            half_extents = [axis / 2.0 for axis in geometry.size]
            return self.p.GEOM_BOX, half_extents, 0.0
        if isinstance(geometry, CylinderGeometry):
            return self.p.GEOM_CYLINDER, [geometry.radius, geometry.height], 0.0
        if isinstance(geometry, PlaneGeometry):
            return self.p.GEOM_BOX, [geometry.size[0] / 2.0, geometry.size[1] / 2.0, 0.001], 0.0
        return self.p.GEOM_BOX, [0.01, 0.01, 0.01], 0.0

    def _create_body(self, object_id: str, mass: float, collision_enabled: bool) -> int:
        obj = self.world_model.get_object(object_id)
        world_pose = self.world.object_world_poses[object_id]
        shape_type, dims, radius = self.geometry_shape(obj.geometry)

        if shape_type == self.p.GEOM_BOX:
            collision_shape = self.p.createCollisionShape(
                self.p.GEOM_BOX,
                halfExtents=dims,
                physicsClientId=self.client_id,
            )
            visual_shape = self.p.createVisualShape(
                self.p.GEOM_BOX,
                halfExtents=dims,
                rgbaColor=[0.7, 0.7, 0.7, 1.0],
                physicsClientId=self.client_id,
            )
        elif shape_type == self.p.GEOM_CYLINDER:
            collision_shape = self.p.createCollisionShape(
                self.p.GEOM_CYLINDER,
                radius=dims[0],
                height=dims[1],
                physicsClientId=self.client_id,
            )
            visual_shape = self.p.createVisualShape(
                self.p.GEOM_CYLINDER,
                radius=dims[0],
                length=dims[1],
                rgbaColor=[0.8, 0.6, 0.3, 1.0],
                physicsClientId=self.client_id,
            )
        else:
            raise ValueError(f"unsupported shape type for object '{object_id}'")

        body_id = self.p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=world_pose["position"],
            baseOrientation=world_pose["orientation"],
            physicsClientId=self.client_id,
        )
        if not collision_enabled:
            self.p.setCollisionFilterGroupMask(body_id, -1, 0, 0, physicsClientId=self.client_id)
        return body_id

    def _build_world_bodies(self) -> None:
        for object_id, obj in self.world_model.objects.items():
            mass = 0.2 if obj.movable else 0.0
            self.body_ids[object_id] = self._create_body(
                object_id,
                mass=mass,
                collision_enabled=obj.collision_enabled,
            )
        logger.debug("Created PyBullet bodies: %s", self.body_ids)

    def _build_tcp_proxy(self) -> None:
        if self.world.tcp_pose_world is None:
            self.world.tcp_pose_world = self.resolve_pose_to_world(self.world_model.robot_state.tcp_pose)
        half_extents = [axis / 2.0 for axis in self.gripper_size]
        collision_shape = self.p.createCollisionShape(
            self.p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self.client_id,
        )
        visual_shape = self.p.createVisualShape(
            self.p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.2, 0.4, 0.9, 0.7],
            physicsClientId=self.client_id,
        )
        self.tcp_body_id = self.p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.world.tcp_pose_world["position"],
            baseOrientation=self.world.tcp_pose_world["orientation"],
            physicsClientId=self.client_id,
        )

    def _set_tcp_pose_world(self, pose_world: WorldPose) -> None:
        self.world.tcp_pose_world = pose_world
        self.p.resetBasePositionAndOrientation(
            self.tcp_body_id,
            pose_world["position"],
            pose_world["orientation"],
            physicsClientId=self.client_id,
        )
        if self.world.attached_object is not None:
            self.update_object_world_pose(self.world.attached_object, pose_world)
            attached_body_id = self.body_ids[self.world.attached_object]
            self.p.resetBasePositionAndOrientation(
                attached_body_id,
                pose_world["position"],
                pose_world["orientation"],
                physicsClientId=self.client_id,
            )

    def _step_simulation(self, num_steps: int = 4) -> None:
        for _ in range(num_steps):
            self.p.stepSimulation(physicsClientId=self.client_id)

    def _interpolate_pose(self, start_pose: WorldPose, end_pose: WorldPose, ratio: float) -> WorldPose:
        start_position = start_pose["position"]
        end_position = end_pose["position"]
        return {
            "frame": end_pose["frame"],
            "position": [
                start_position[i] + (end_position[i] - start_position[i]) * ratio
                for i in range(3)
            ],
            "orientation": list(end_pose["orientation"]),
        }

    def _collision_query(self, excluded_objects: set[str], moving_entities: List[str], step_type: PrimitiveType) -> Dict[str, Any] | None:
        for object_id, body_id in self.body_ids.items():
            if object_id in excluded_objects:
                continue
            contacts = self.p.getClosestPoints(
                bodyA=self.tcp_body_id,
                bodyB=body_id,
                distance=0.0,
                physicsClientId=self.client_id,
            )
            if contacts:
                logger.debug(
                    "PyBullet contact step=%s moving=%s obstacle=%s contacts=%d",
                    step_type.value,
                    moving_entities,
                    object_id,
                    len(contacts),
                )
                return self.build_collision_error(step_type, moving_entities, object_id)

        attached_object = self.world.attached_object
        if attached_object is not None:
            attached_body_id = self.body_ids[attached_object]
            for object_id, body_id in self.body_ids.items():
                if object_id in excluded_objects or object_id == attached_object:
                    continue
                contacts = self.p.getClosestPoints(
                    bodyA=attached_body_id,
                    bodyB=body_id,
                    distance=0.0,
                    physicsClientId=self.client_id,
                )
                if contacts:
                    return self.build_collision_error(step_type, [attached_object], object_id)
        return None

    def _move_tcp_linearly(
        self,
        target_pose_world: WorldPose,
        *,
        step_type: PrimitiveType,
        excluded_objects: set[str],
        moving_entities: List[str],
    ) -> Dict[str, Any] | None:
        start_pose = self.world.tcp_pose_world or target_pose_world
        for sample_index in range(1, self.path_samples + 1):
            ratio = sample_index / self.path_samples
            sample_pose = self._interpolate_pose(start_pose, target_pose_world, ratio)
            self._set_tcp_pose_world(sample_pose)
            self._step_simulation()
            collision = self._collision_query(excluded_objects, moving_entities, step_type)
            if collision is not None:
                return collision
        return None

    def _attach_object(self, object_id: str) -> None:
        body_id = self.body_ids[object_id]
        if self.grasp_constraint_id is not None:
            self.p.removeConstraint(self.grasp_constraint_id, physicsClientId=self.client_id)
        self.p.changeDynamics(body_id, -1, mass=0.001, physicsClientId=self.client_id)
        self.p.setCollisionFilterPair(self.tcp_body_id, body_id, -1, -1, 0, physicsClientId=self.client_id)
        self.grasp_constraint_id = self.p.createConstraint(
            parentBodyUniqueId=self.tcp_body_id,
            parentLinkIndex=-1,
            childBodyUniqueId=body_id,
            childLinkIndex=-1,
            jointType=self.p.JOINT_FIXED,
            jointAxis=[0.0, 0.0, 0.0],
            parentFramePosition=[0.0, 0.0, -self.gripper_half_height],
            childFramePosition=[0.0, 0.0, 0.0],
            physicsClientId=self.client_id,
        )
        self._step_simulation()

    def _detach_object(self, object_id: str) -> None:
        body_id = self.body_ids[object_id]
        if self.grasp_constraint_id is not None:
            self.p.removeConstraint(self.grasp_constraint_id, physicsClientId=self.client_id)
            self.grasp_constraint_id = None
        obj = self.world_model.get_object(object_id)
        self.p.changeDynamics(body_id, -1, mass=0.2 if obj.movable else 0.0, physicsClientId=self.client_id)
        self.p.setCollisionFilterPair(self.tcp_body_id, body_id, -1, -1, 1, physicsClientId=self.client_id)
        self._step_simulation()

    def execute_step(self, step: ActionStep) -> Tuple[bool, Dict[str, Any] | None]:
        parsed_inputs = step.parsed_inputs()
        step_type = step.type

        if step_type == PrimitiveType.FIND_OBJECT:
            return super().execute_step(step)

        if step_type == PrimitiveType.APPROACH:
            if parsed_inputs.target_object not in self.world.bound_objects:
                return super().execute_step(step)
            target_pose_world = self.resolve_pose_ref_to_world(parsed_inputs.approach_pose)
            excluded = {parsed_inputs.target_object}
            owner = self.infer_owner_for_ref(parsed_inputs.approach_pose.ref)
            if owner is not None:
                excluded.add(owner)
            collision = self._move_tcp_linearly(
                target_pose_world,
                step_type=step_type,
                excluded_objects=excluded,
                moving_entities=["gripper_tcp"],
            )
            if collision is not None:
                return False, collision
            self.world.achieved_effects.add(
                f"tcp_at_pregrasp_pose:{parsed_inputs.target_feature}"
                if parsed_inputs.target_feature
                else "tcp_at_pregrasp_pose"
            )
            return True, None

        if step_type == PrimitiveType.GRASP:
            ok, err = super().execute_step(step)
            if ok and self.world.attached_object is not None:
                self._attach_object(self.world.attached_object)
            return ok, err

        if step_type == PrimitiveType.MOVE_LINEAR:
            target_pose_world = self.resolve_pose_ref_to_world(parsed_inputs.target_pose)
            owner = self.infer_owner_for_ref(parsed_inputs.target_pose.ref)
            excluded = set()
            if self.world.attached_object is not None:
                excluded.add(self.world.attached_object)
            if owner is not None:
                excluded.add(owner)
            collision = self._move_tcp_linearly(
                target_pose_world,
                step_type=step_type,
                excluded_objects=excluded,
                moving_entities=["gripper_tcp"],
            )
            if collision is not None:
                return False, collision
            return True, None

        if step_type == PrimitiveType.MOVE_JOINT:
            logger.debug("MOVE_JOINT is approximated as a no-op in PyBullet demo backend")
            return True, None

        if step_type == PrimitiveType.PLACE:
            ok, err = super().execute_step(step)
            if ok and self.world.tcp_pose_world is not None:
                self.p.resetBasePositionAndOrientation(
                    self.body_ids[parsed_inputs.target_object],
                    self.world.tcp_pose_world["position"],
                    self.world.tcp_pose_world["orientation"],
                    physicsClientId=self.client_id,
                )
                self._step_simulation()
            return ok, err

        if step_type == PrimitiveType.RELEASE:
            if self.world.attached_object == parsed_inputs.target_object:
                self._detach_object(parsed_inputs.target_object)
            return super().execute_step(step)

        if step_type == PrimitiveType.INSERT:
            geometry_collision = self.check_insert_geometry_collision(parsed_inputs)
            if geometry_collision is not None:
                return False, geometry_collision
            target_pose_world = (
                self.resolve_feature_pose_to_world(parsed_inputs.target_feature)
                if parsed_inputs.target_feature
                else None
            )
            if target_pose_world is None:
                return super().execute_step(step)
            collision = self._move_tcp_linearly(
                target_pose_world,
                step_type=step_type,
                excluded_objects={parsed_inputs.source_object, parsed_inputs.target_object},
                moving_entities=["gripper_tcp", parsed_inputs.source_object],
            )
            if collision is not None:
                return False, collision
            return super().execute_step(step)

        return super().execute_step(step)


def step_inputs_frame_candidates(step: ActionStep) -> List[str]:
    refs: List[str] = []
    if "target_pose" in step.inputs and isinstance(step.inputs["target_pose"], dict):
        ref = step.inputs["target_pose"].get("ref")
        if ref:
            refs.append(ref)
    if "approach_pose" in step.inputs and isinstance(step.inputs["approach_pose"], dict):
        ref = step.inputs["approach_pose"].get("ref")
        if ref:
            refs.append(ref)
    if "destination_pose" in step.inputs and isinstance(step.inputs["destination_pose"], dict):
        ref = step.inputs["destination_pose"].get("ref")
        if ref:
            refs.append(ref)
    return refs


def step_input_feature_candidates(step: ActionStep) -> List[str]:
    features: List[str] = []
    for key in ["target_feature", "source_feature"]:
        value = step.inputs.get(key)
        if isinstance(value, str) and value:
            features.append(value)
    return features


def validate_step_refs(
    ir: GenericCobotIR,
    step: ActionStep,
    world_model: WorldModel | None = None,
) -> Tuple[bool, Dict[str, Any] | None]:
    known_frames = set(ir.world_binding.frames.keys())
    known_features = set(ir.world_binding.features.keys())

    world_frame_registry_paths = set()
    if world_model is not None:
        known_frames.update(world_model.frames.keys())
        known_features.update(world_model.features.keys())
        world_frame_registry_paths = {frame.registry_path for frame in world_model.frames.values()}

    ir_frame_registry_values = set(ir.world_binding.frames.values())

    for ref in step_inputs_frame_candidates(step):
        if ref not in known_frames and ref not in world_frame_registry_paths and ref not in ir_frame_registry_values:
            logger.debug("Step %s has unknown frame reference: %s", step.step_id, ref)
            return False, {
                "error_id": "e_unknown_frame_ref",
                "type": "binding_error",
                "message": f"unknown frame reference: {ref}",
                "entities": [ref],
                "suggested_repairs": ["fix_world_binding_frames", "fix_pose_ref"],
            }

    for feature in step_input_feature_candidates(step):
        if feature not in known_features:
            logger.debug("Step %s has unknown feature reference: %s", step.step_id, feature)
            return False, {
                "error_id": "e_unknown_feature_ref",
                "type": "binding_error",
                "message": f"unknown feature reference: {feature}",
                "entities": [feature],
                "suggested_repairs": ["declare_feature_in_world_binding", "fix_feature_ref"],
            }

    return True, None


def check_goal_reached(ir: GenericCobotIR, sim: MockSimulator) -> bool:
    for cond in ir.task_spec.success_condition:
        if cond == "no_collision":
            continue
        if cond not in sim.world.achieved_effects:
            return False
    return True


def run_ir(
    ir: GenericCobotIR,
    world_model: WorldModel | None = None,
    *,
    simulator_backend: str = "mock",
    pybullet_gui: bool = False,
) -> VerificationResult:
    if simulator_backend == "mock":
        sim: MockSimulator = MockSimulator(ir, world_model=world_model)
    elif simulator_backend == "pybullet":
        sim = PyBulletSimulator(
            ir,
            world_model=world_model,
            gui=pybullet_gui,
        )
    else:
        raise ValueError(f"unknown simulator_backend: {simulator_backend}")
    executed = 0
    collision_detected = False
    traces: List[StepTrace] = []
    try:
        for step in ir.action_plan:
            before_state = sim.snapshot_state()
            logger.debug("Step %s before_state=%s", step.step_id, before_state)
            ok_ref, ref_err = validate_step_refs(ir, step, world_model=world_model)
            if not ok_ref:
                step.status = StepStatus.FAILED
                traces.append(
                    StepTrace(
                        step_id=step.step_id,
                        step_type=step.type.value,
                        before_state=before_state,
                        after_state=deepcopy(before_state),
                        success=False,
                        error=ref_err,
                    )
                )
                return VerificationResult(
                    task_id=ir.task_id,
                    simulator=sim.simulator_name,
                    status="failed",
                    summary=VerificationSummary(
                        steps_total=len(ir.action_plan),
                        steps_executed=executed,
                        goal_reached=False,
                        collision_detected=False,
                    ),
                    errors=[
                        VerificationError(
                            error_id=ref_err["error_id"],
                            step_id=step.step_id,
                            type=ref_err["type"],
                            severity=ErrorSeverity.HIGH,
                            message=ref_err["message"],
                            entities=ref_err.get("entities", []),
                            suggested_repairs=ref_err.get("suggested_repairs", []),
                        )
                    ],
                    metrics=VerificationMetrics(
                        min_clearance=0.01,
                        max_joint_velocity_ratio=0.4,
                        estimated_execution_time_sec=float(executed + 1),
                    ),
                    traces=traces,
                )

            ok, err = sim.execute_step(step)
            after_state = sim.snapshot_state()
            logger.debug("Step %s after_state=%s success=%s error=%s", step.step_id, after_state, ok, err)
            traces.append(
                StepTrace(
                    step_id=step.step_id,
                    step_type=step.type.value,
                    before_state=before_state,
                    after_state=after_state,
                    success=ok,
                    error=err,
                )
            )

            if not ok:
                step.status = StepStatus.FAILED
                if err["type"] == "collision":
                    collision_detected = True

                return VerificationResult(
                    task_id=ir.task_id,
                    simulator=sim.simulator_name,
                    status="failed",
                    summary=VerificationSummary(
                        steps_total=len(ir.action_plan),
                        steps_executed=executed + 1,
                        goal_reached=False,
                        collision_detected=collision_detected,
                    ),
                    errors=[
                        VerificationError(
                            error_id=err["error_id"],
                            step_id=step.step_id,
                            type=err["type"],
                            severity=ErrorSeverity.HIGH,
                            message=err["message"],
                            entities=err.get("entities", []),
                            suggested_repairs=err.get("suggested_repairs", []),
                        )
                    ],
                    metrics=VerificationMetrics(
                        min_clearance=0.0015 if collision_detected else 0.01,
                        max_joint_velocity_ratio=0.58 if collision_detected else 0.4,
                        estimated_execution_time_sec=float(executed + 1),
                    ),
                    traces=traces,
                )

            step.status = StepStatus.SIMULATED
            executed += 1

        goal_reached = check_goal_reached(ir, sim)
        logger.debug("Execution completed task_id=%s steps=%d goal_reached=%s", ir.task_id, executed, goal_reached)

        return VerificationResult(
            task_id=ir.task_id,
            simulator=sim.simulator_name,
            status="passed" if goal_reached else "failed",
            summary=VerificationSummary(
                steps_total=len(ir.action_plan),
                steps_executed=executed,
                goal_reached=goal_reached,
                collision_detected=False,
            ),
            errors=[] if goal_reached else [
                VerificationError(
                    error_id="e_goal_not_reached",
                    step_id=ir.action_plan[-1].step_id if ir.action_plan else "none",
                    type="goal_not_reached",
                    severity=ErrorSeverity.MEDIUM,
                    message="task completed without runtime error but success condition was not satisfied",
                    entities=ir.task_spec.success_condition,
                    suggested_repairs=["add_check_step", "refine_expected_effects"],
                )
            ],
            metrics=VerificationMetrics(
                min_clearance=0.01,
                max_joint_velocity_ratio=0.4,
                estimated_execution_time_sec=float(executed),
            ),
            traces=traces,
        )
    finally:
        if hasattr(sim, "close"):
            sim.close()
