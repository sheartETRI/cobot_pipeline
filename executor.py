# executor.py (updated)
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Optional

from world_model import BoxGeometry, CylinderGeometry, ObjectModel, PlaneGeometry, WorldModel

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


@dataclass
class MockWorldState:
    bound_objects: set[str] = field(default_factory=set)
    known_features: set[str] = field(default_factory=set)
    attached_object: str | None = None
    achieved_effects: set[str] = field(default_factory=set)


@dataclass
class AABB:
    center: Tuple[float, float, float]
    half_extents: Tuple[float, float, float]


class MockSimulator:
    """
    연구 초기용 mock simulator.
    실제 물리엔진 대신 단순 규칙으로 success/failure를 반환함.
    world model이 주어지면 초기 상태를 추가 반영함.
    """

    def __init__(self, ir: GenericCobotIR, world_model: WorldModel | None = None) -> None:
        self.world_model = world_model
        known_features = set(ir.world_binding.features.keys())
        bound_objects: set[str] = set()
        attached_object: Optional[str] = None

        if world_model is not None:
            # features from world
            known_features.update(world_model.features.keys())

            # bound objects: include world objects
            bound_objects.update(world_model.objects.keys())

            # Explicit registry_id mapping between IR object bindings and world objects.
            for alias, registry_id in ir.world_binding.objects.items():
                matched = world_model.find_object_by_registry_id(registry_id)
                if matched is not None and matched.object_id == alias:
                    bound_objects.add(alias)
                    logger.debug("Bound world object by registry_id: alias=%s registry_id=%s", alias, registry_id)
            # attached object
            attached_object = world_model.robot_state.attached_object

        self.world = MockWorldState(
            bound_objects=bound_objects,
            known_features=known_features,
            attached_object=attached_object,
        )
        self.simulator_name = "mock_sim"
        logger.debug(
            "Initialized MockSimulator with bound_objects=%s known_features=%s attached_object=%s",
            sorted(bound_objects),
            sorted(known_features),
            attached_object,
        )

    def _geometry_half_extents(self, obj: ObjectModel) -> Tuple[float, float, float]:
        geometry = obj.geometry
        if isinstance(geometry, BoxGeometry):
            return (
                geometry.size[0] / 2.0,
                geometry.size[1] / 2.0,
                geometry.size[2] / 2.0,
            )
        if isinstance(geometry, CylinderGeometry):
            return (
                geometry.radius,
                geometry.radius,
                geometry.height / 2.0,
            )
        if isinstance(geometry, PlaneGeometry):
            return (
                geometry.size[0] / 2.0,
                geometry.size[1] / 2.0,
                0.001,
            )
        return (0.01, 0.01, 0.01)

    def _vector_add(
        self,
        lhs: Tuple[float, float, float],
        rhs: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        return (lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2])

    def _resolve_object_center(self, object_id: str) -> Optional[Tuple[float, float, float]]:
        if self.world_model is None or not self.world_model.has_object(object_id):
            return None
        pose = self.world_model.get_object(object_id).pose
        return self._resolve_pose_center(pose.frame, tuple(pose.position))

    def _resolve_feature_center(self, feature_id: str) -> Optional[Tuple[float, float, float]]:
        if self.world_model is None or not self.world_model.has_feature(feature_id):
            return None
        feature = self.world_model.get_feature(feature_id)
        parent_center = self._resolve_object_center(feature.parent_object)
        if parent_center is None:
            return None
        return self._vector_add(parent_center, tuple(feature.local_pose.position))

    def _resolve_frame_center(self, frame_id: str) -> Optional[Tuple[float, float, float]]:
        if self.world_model is None or not self.world_model.has_frame(frame_id):
            return None
        frame = self.world_model.get_frame(frame_id)
        return self._resolve_pose_center(frame.pose.frame, tuple(frame.pose.position))

    def _resolve_pose_center(
        self,
        ref_name: str,
        offset: Tuple[float, float, float],
    ) -> Optional[Tuple[float, float, float]]:
        if self.world_model is None:
            return None
        if ref_name == self.world_model.world_frame:
            return offset
        if self.world_model.has_object(ref_name):
            base = self._resolve_object_center(ref_name)
            return self._vector_add(base, offset) if base is not None else None
        if self.world_model.has_feature(ref_name):
            base = self._resolve_feature_center(ref_name)
            return self._vector_add(base, offset) if base is not None else None
        if self.world_model.has_frame(ref_name):
            base = self._resolve_frame_center(ref_name)
            return self._vector_add(base, offset) if base is not None else None

        for frame in self.world_model.frames.values():
            if frame.registry_path == ref_name:
                base = self._resolve_frame_center(frame.frame_id)
                return self._vector_add(base, offset) if base is not None else None
        return None

    def _resolve_pose_ref_center(self, pose_ref: Any) -> Optional[Tuple[float, float, float]]:
        offset = tuple(getattr(pose_ref, "offset", [0.0, 0.0, 0.0]))
        return self._resolve_pose_center(pose_ref.ref, offset)

    def _infer_object_owner_for_ref(self, ref_name: str) -> Optional[str]:
        if self.world_model is None:
            return None
        if self.world_model.has_object(ref_name):
            return ref_name
        if self.world_model.has_feature(ref_name):
            return self.world_model.get_feature(ref_name).parent_object
        if self.world_model.has_frame(ref_name):
            registry_path = self.world_model.get_frame(ref_name).registry_path
            registry_base = registry_path.split("/")[0]
            matched = self.world_model.find_object_by_registry_id(registry_base)
            return matched.object_id if matched is not None else None
        for frame in self.world_model.frames.values():
            if frame.registry_path == ref_name:
                registry_base = frame.registry_path.split("/")[0]
                matched = self.world_model.find_object_by_registry_id(registry_base)
                return matched.object_id if matched is not None else None
        return None

    def _object_aabb(self, object_id: str) -> Optional[AABB]:
        if self.world_model is None or not self.world_model.has_object(object_id):
            return None
        obj = self.world_model.get_object(object_id)
        center = self._resolve_object_center(object_id)
        if center is None:
            return None
        return AABB(center=center, half_extents=self._geometry_half_extents(obj))

    def _aabb_overlaps(self, lhs: AABB, rhs: AABB, clearance: float = 0.0) -> bool:
        for i in range(3):
            if abs(lhs.center[i] - rhs.center[i]) > (lhs.half_extents[i] + rhs.half_extents[i] + clearance):
                return False
        return True

    def _check_pose_collision(
        self,
        center: Optional[Tuple[float, float, float]],
        half_extents: Tuple[float, float, float],
        excluded_objects: set[str],
        clearance: float = 0.0,
    ) -> Dict[str, Any] | None:
        if self.world_model is None or center is None:
            return None

        probe = AABB(center=center, half_extents=half_extents)
        logger.debug(
            "Checking pose collision: center=%s half_extents=%s excluded=%s clearance=%.4f",
            center,
            half_extents,
            sorted(excluded_objects),
            clearance,
        )
        for object_id, obj in self.world_model.objects.items():
            if not obj.collision_enabled or object_id in excluded_objects:
                continue
            obstacle = self._object_aabb(object_id)
            if obstacle is None:
                continue
            if self._aabb_overlaps(probe, obstacle, clearance=clearance):
                logger.debug(
                    "AABB collision detected with object=%s probe=%s obstacle=%s",
                    object_id,
                    probe,
                    obstacle,
                )
                return {
                    "error_id": "e_geometry_collision",
                    "type": "collision",
                    "message": f"predicted AABB collision with object '{object_id}'",
                    "entities": ["gripper_tcp", object_id],
                    "suggested_repairs": [
                        "increase_clearance",
                        "replan_path",
                        "adjust_target_pose",
                    ],
                }
        return None

    def _check_insert_geometry_collision(self, step_inputs: Any) -> Dict[str, Any] | None:
        if self.world_model is None:
            return None
        if not self.world_model.has_object(step_inputs.source_object):
            return None
        if not step_inputs.target_feature or not self.world_model.has_feature(step_inputs.target_feature):
            return None

        source_object = self.world_model.get_object(step_inputs.source_object)
        target_feature = self.world_model.get_feature(step_inputs.target_feature)
        source_extents = self._geometry_half_extents(source_object)
        source_diameter = max(source_extents[0], source_extents[1]) * 2.0
        feature_width = target_feature.width
        feature_depth = target_feature.depth
        logger.debug(
            "Checking insert geometry: source=%s diameter=%.4f target_feature=%s width=%s depth=%s insert_depth=%.4f",
            step_inputs.source_object,
            source_diameter,
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
            approach_center = self._resolve_pose_ref_center(parsed_inputs.approach_pose)
            excluded = {target}
            owner = self._infer_object_owner_for_ref(parsed_inputs.approach_pose.ref)
            if owner is not None:
                excluded.add(owner)
            collision = self._check_pose_collision(
                center=approach_center,
                half_extents=(0.01, 0.01, 0.02),
                excluded_objects=excluded,
                clearance=getattr(parsed_constraints, "min_clearance", 0.0),
            )
            if collision is not None:
                logger.debug("approach failed due to predicted collision: %s", collision)
                return False, collision
            if parsed_inputs.target_feature:
                self.world.achieved_effects.add(
                    f"tcp_at_pregrasp_pose:{parsed_inputs.target_feature}"
                )
            else:
                self.world.achieved_effects.add("tcp_at_pregrasp_pose")
            logger.debug("approach succeeded for target=%s", target)
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
            self.world.achieved_effects.add(f"object_attached:{target}")
            if parsed_inputs.target_feature:
                self.world.achieved_effects.add(
                    f"grasp_target_feature:{target}:{parsed_inputs.target_feature}"
                )
            logger.debug("grasp succeeded for target=%s", target)
            return True, None

        if step_type == PrimitiveType.RETREAT:
            return True, None

        if step_type == PrimitiveType.MOVE_LINEAR:
            target_center = self._resolve_pose_ref_center(parsed_inputs.target_pose)
            clearance = 0.005 if getattr(parsed_constraints, "collision_avoidance", False) else 0.0
            owner = self._infer_object_owner_for_ref(parsed_inputs.target_pose.ref)
            excluded = {self.world.attached_object} if self.world.attached_object else set()
            if owner is not None:
                excluded.add(owner)
            collision = self._check_pose_collision(
                center=target_center,
                half_extents=(0.01, 0.01, 0.02),
                excluded_objects=excluded,
                clearance=clearance,
            )
            if collision is not None:
                logger.debug("move_linear failed due to predicted collision: %s", collision)
                return False, collision
            logger.debug("move_linear succeeded to ref=%s", parsed_inputs.target_pose.ref)
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
            target_center = self._resolve_pose_ref_center(parsed_inputs.destination_pose)
            owner = self._infer_object_owner_for_ref(parsed_inputs.destination_pose.ref)
            excluded = {target}
            if owner is not None:
                excluded.add(owner)
            target_aabb = self._object_aabb(target)
            half_extents = target_aabb.half_extents if target_aabb is not None else (0.02, 0.02, 0.02)
            collision = self._check_pose_collision(
                center=target_center,
                half_extents=half_extents,
                excluded_objects=excluded,
                clearance=getattr(parsed_constraints, "placement_tolerance", 0.0),
            )
            if collision is not None:
                logger.debug("place failed due to predicted collision: %s", collision)
                return False, collision
            effect_suffix = (
                f"{target}:{parsed_inputs.target_feature}"
                if parsed_inputs.target_feature
                else target
            )
            self.world.achieved_effects.add(f"object_at_destination:{effect_suffix}")
            logger.debug("place succeeded for target=%s effect=%s", target, effect_suffix)
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
            self.world.achieved_effects.add(
                f"objects_aligned:{source_suffix}:{target_suffix}"
            )
            logger.debug("align succeeded source=%s target=%s", source_suffix, target_suffix)
            return True, None

        if step_type == PrimitiveType.INSERT:
            speed = getattr(parsed_constraints, "speed", 0.03)
            max_force = getattr(parsed_constraints, "max_force", None)

            geometry_collision = self._check_insert_geometry_collision(parsed_inputs)
            if geometry_collision is not None:
                logger.debug("insert failed due to geometry collision: %s", geometry_collision)
                return False, geometry_collision

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

            target_suffix = (
                parsed_inputs.target_feature
                if parsed_inputs.target_feature
                else parsed_inputs.target_object
            )
            self.world.achieved_effects.add(
                f"{parsed_inputs.source_object}_inserted_into_{target_suffix}"
            )
            logger.debug("insert succeeded source=%s target=%s", parsed_inputs.source_object, target_suffix)
            return True, None

        if step_type == PrimitiveType.RELEASE:
            target = parsed_inputs.target_object
            if self.world.attached_object == target:
                self.world.attached_object = None
            self.world.achieved_effects.add("release_completed")
            self.world.achieved_effects.add(f"object_detached:{target}")
            logger.debug("release succeeded for target=%s", target)
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
            logger.debug("check succeeded for condition=%s", condition)
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
    feature_keys = [
        "target_feature",
        "source_feature",
    ]
    features: List[str] = []
    for key in feature_keys:
        value = step.inputs.get(key)
        if isinstance(value, str) and value:
            features.append(value)
    return features


def validate_step_refs(ir: GenericCobotIR, step: ActionStep, world_model: WorldModel | None = None) -> Tuple[bool, Dict[str, Any] | None]:
    """
    Validate frames and features used by a step.
    Accepts world_model to allow validation against world-defined frames/features.
    """
    known_frames = set(ir.world_binding.frames.keys())
    known_features = set(ir.world_binding.features.keys())

    # include world-defined frames/features if provided
    world_frame_registry_paths = set()
    if world_model is not None:
        known_frames.update(world_model.frames.keys())
        known_features.update(world_model.features.keys())
        world_frame_registry_paths = {f.registry_path for f in world_model.frames.values()}

    # include IR->registry values as acceptable frame references (fallback)
    ir_frame_registry_values = set(ir.world_binding.frames.values())

    for ref in step_inputs_frame_candidates(step):
        # accepted if:
        #  - ref is IR frame alias
        #  - ref is world frame id
        #  - ref equals one of IR frame registry values (like "obj_xxx/frame")
        #  - ref equals or matches start of a world frame registry_path
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


def run_ir(ir: GenericCobotIR, world_model: WorldModel | None = None) -> VerificationResult:
    sim = MockSimulator(ir, world_model=world_model)
    executed = 0
    collision_detected = False
    traces: List[StepTrace] = []

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
