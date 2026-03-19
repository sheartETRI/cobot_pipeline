from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from typing import Any, Dict, List, Tuple

from world_model import WorldModel

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


@dataclass
class MockWorldState:
    bound_objects: set[str] = field(default_factory=set)
    known_features: set[str] = field(default_factory=set)
    attached_object: str | None = None
    achieved_effects: set[str] = field(default_factory=set)


class MockSimulator:
    """
    연구 초기용 mock simulator.
    실제 물리엔진 대신 단순 규칙으로 success/failure를 반환함.
    world model이 주어지면 초기 상태를 추가 반영함.
    """

    def __init__(self, ir: GenericCobotIR, world_model: WorldModel | None = None) -> None:
        known_features = set(ir.world_binding.features.keys())
        if world_model is not None:
            known_features.update(world_model.features.keys())

        self.world = MockWorldState(
            bound_objects=set(world_model.objects.keys()) if world_model is not None else set(),
            known_features=known_features,
            attached_object=(world_model.robot_state.attached_object if world_model is not None else None),
        )
        self.simulator_name = "mock_sim"

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

        if step_type == PrimitiveType.FIND_OBJECT:
            obj_name = parsed_inputs.object
            self.world.bound_objects.add(obj_name)
            self.world.achieved_effects.add(f"object_pose_bound:{obj_name}")
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
            if parsed_inputs.target_feature:
                self.world.achieved_effects.add(
                    f"tcp_at_pregrasp_pose:{parsed_inputs.target_feature}"
                )
            else:
                self.world.achieved_effects.add("tcp_at_pregrasp_pose")
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
            return True, None

        if step_type == PrimitiveType.RETREAT:
            return True, None

        if step_type == PrimitiveType.MOVE_LINEAR:
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
            effect_suffix = (
                f"{target}:{parsed_inputs.target_feature}"
                if parsed_inputs.target_feature
                else target
            )
            self.world.achieved_effects.add(f"object_at_destination:{effect_suffix}")
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
            return True, None

        if step_type == PrimitiveType.INSERT:
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

            target_suffix = (
                parsed_inputs.target_feature
                if parsed_inputs.target_feature
                else parsed_inputs.target_object
            )
            self.world.achieved_effects.add(
                f"{parsed_inputs.source_object}_inserted_into_{target_suffix}"
            )
            return True, None

        if step_type == PrimitiveType.RELEASE:
            target = parsed_inputs.target_object
            if self.world.attached_object == target:
                self.world.attached_object = None
            self.world.achieved_effects.add("release_completed")
            self.world.achieved_effects.add(f"object_detached:{target}")
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


def validate_step_refs(ir: GenericCobotIR, step: ActionStep) -> Tuple[bool, Dict[str, Any] | None]:
    known_frames = set(ir.world_binding.frames.keys())
    known_features = set(ir.world_binding.features.keys())

    for ref in step_inputs_frame_candidates(step):
        if ref not in known_frames:
            return False, {
                "error_id": "e_unknown_frame_ref",
                "type": "binding_error",
                "message": f"unknown frame reference: {ref}",
                "entities": [ref],
                "suggested_repairs": ["fix_world_binding_frames", "fix_pose_ref"],
            }

    for feature in step_input_feature_candidates(step):
        if feature not in known_features:
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
        ok_ref, ref_err = validate_step_refs(ir, step)
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
