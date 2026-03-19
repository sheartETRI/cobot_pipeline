from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict, field_validator


class StepStatus(str, Enum):
    PLANNED = "planned"
    SIMULATED = "simulated"
    FAILED = "failed"
    VERIFIED = "verified"


class PriorityLevel(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class ErrorSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RepairOp(str, Enum):
    REPLACE = "replace"
    APPEND = "append"
    REMOVE = "remove"


class PrimitiveType(str, Enum):
    FIND_OBJECT = "find_object"
    MOVE_JOINT = "move_joint"
    MOVE_LINEAR = "move_linear"
    APPROACH = "approach"
    RETREAT = "retreat"
    GRASP = "grasp"
    RELEASE = "release"
    PLACE = "place"
    ALIGN = "align"
    INSERT = "insert"
    WAIT = "wait"
    CHECK = "check"


class PoseRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref: str
    offset: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation_policy: str

    @field_validator("offset")
    @classmethod
    def validate_offset(cls, v: List[float]) -> List[float]:
        if len(v) != 3:
            raise ValueError("offset must have exactly 3 elements")
        return v


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str
    command_text: str
    priority: PriorityLevel = PriorityLevel.NORMAL
    success_condition: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)


class RobotProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    robot_type: str = "generic_cobot"
    arm_dof: int = Field(..., ge=1, le=20)
    has_gripper: bool = True
    tool_frame: str
    base_frame: str
    motion_limits_profile: str = "default_cobot"


class FeatureBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parent_object: str = Field(..., description="Object alias that owns this feature")
    feature_type: str = Field(..., description="Feature type such as hole, slot, surface")
    frame: Optional[str] = Field(default=None, description="Frame alias for this feature")
    description: Optional[str] = None


class WorldBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    objects: Dict[str, str] = Field(default_factory=dict)
    frames: Dict[str, str] = Field(default_factory=dict)
    regions: Dict[str, str] = Field(default_factory=dict)
    features: Dict[str, FeatureBinding] = Field(default_factory=dict)

    @field_validator("features")
    @classmethod
    def validate_feature_bindings(cls, features: Dict[str, FeatureBinding]) -> Dict[str, FeatureBinding]:
        for feature_name, binding in features.items():
            if not feature_name:
                raise ValueError("feature names must be non-empty")
            if not binding.feature_type:
                raise ValueError(f"feature '{feature_name}' must declare feature_type")
        return features


class VerificationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collision_check: bool = True
    ik_check: bool = True
    joint_limit_check: bool = True
    velocity_limit_check: bool = True
    force_limit_check: bool = False
    max_retry: int = Field(3, ge=0, le=20)
    acceptance_rules: List[str] = Field(default_factory=list)


class RepairHistoryItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    reason: str
    patch_summary: str


class RepairState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retry_count: int = Field(0, ge=0)
    last_error: Optional[str] = None
    repair_history: List[RepairHistoryItem] = Field(default_factory=list)


class FindObjectInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    object: str


class MoveJointInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_joint_ref: str


class MoveLinearInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_pose: PoseRef


class ApproachInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_object: str
    target_feature: Optional[str] = None
    approach_pose: PoseRef


class RetreatInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    direction: str
    distance: float = Field(..., gt=0.0)


class GraspInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_object: str
    target_feature: Optional[str] = None
    grasp_mode: str


class ReleaseInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_object: str


class PlaceInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_object: str
    target_feature: Optional[str] = None
    destination_pose: PoseRef


class AlignInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_object: str
    source_feature: Optional[str] = None
    target_object: str
    target_feature: Optional[str] = None
    alignment_mode: str


class InsertInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_object: str
    source_feature: Optional[str] = None
    target_object: str
    target_feature: Optional[str] = None
    insert_axis: str
    insert_depth: float = Field(..., gt=0.0)


class WaitInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    duration_sec: float = Field(..., ge=0.0)


class CheckInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    condition: str


class FindObjectConstraints(BaseModel):
    model_config = ConfigDict(extra="allow")
    must_exist: bool = True


class MoveJointConstraints(BaseModel):
    model_config = ConfigDict(extra="allow")
    speed_scale: float = Field(0.5, gt=0.0, le=1.0)
    acc_scale: float = Field(0.5, gt=0.0, le=1.0)


class MoveLinearConstraints(BaseModel):
    model_config = ConfigDict(extra="allow")
    speed: float = Field(0.1, gt=0.0)
    collision_avoidance: bool = True


class ApproachConstraints(BaseModel):
    model_config = ConfigDict(extra="allow")
    min_clearance: float = Field(0.0, ge=0.0)
    speed: float = Field(0.1, gt=0.0)


class RetreatConstraints(BaseModel):
    model_config = ConfigDict(extra="allow")
    speed: float = Field(0.1, gt=0.0)


class GraspConstraints(BaseModel):
    model_config = ConfigDict(extra="allow")
    grip_force: float = Field(10.0, gt=0.0)
    require_contact: bool = True


class ReleaseConstraints(BaseModel):
    model_config = ConfigDict(extra="allow")


class PlaceConstraints(BaseModel):
    model_config = ConfigDict(extra="allow")
    placement_tolerance: float = Field(0.01, gt=0.0)


class AlignConstraints(BaseModel):
    model_config = ConfigDict(extra="allow")
    angular_tolerance_deg: float = Field(5.0, gt=0.0)
    position_tolerance_m: float = Field(0.005, gt=0.0)


class InsertConstraints(BaseModel):
    model_config = ConfigDict(extra="allow")
    speed: float = Field(0.03, gt=0.0)
    max_force: Optional[float] = Field(default=None, gt=0.0)
    require_alignment: bool = True


class WaitConstraints(BaseModel):
    model_config = ConfigDict(extra="allow")


class CheckConstraints(BaseModel):
    model_config = ConfigDict(extra="allow")
    must_be_true: bool = True


class ActionStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    type: PrimitiveType
    inputs: Dict[str, Any]
    constraints: Dict[str, Any] = Field(default_factory=dict)
    expected_effects: List[str] = Field(default_factory=list)
    on_failure: Dict[str, Any] = Field(default_factory=dict)
    status: StepStatus = StepStatus.PLANNED

    def parsed_inputs(self) -> BaseModel:
        mapping = {
            PrimitiveType.FIND_OBJECT: FindObjectInputs,
            PrimitiveType.MOVE_JOINT: MoveJointInputs,
            PrimitiveType.MOVE_LINEAR: MoveLinearInputs,
            PrimitiveType.APPROACH: ApproachInputs,
            PrimitiveType.RETREAT: RetreatInputs,
            PrimitiveType.GRASP: GraspInputs,
            PrimitiveType.RELEASE: ReleaseInputs,
            PrimitiveType.PLACE: PlaceInputs,
            PrimitiveType.ALIGN: AlignInputs,
            PrimitiveType.INSERT: InsertInputs,
            PrimitiveType.WAIT: WaitInputs,
            PrimitiveType.CHECK: CheckInputs,
        }
        return mapping[self.type](**self.inputs)

    def parsed_constraints(self) -> BaseModel:
        mapping = {
            PrimitiveType.FIND_OBJECT: FindObjectConstraints,
            PrimitiveType.MOVE_JOINT: MoveJointConstraints,
            PrimitiveType.MOVE_LINEAR: MoveLinearConstraints,
            PrimitiveType.APPROACH: ApproachConstraints,
            PrimitiveType.RETREAT: RetreatConstraints,
            PrimitiveType.GRASP: GraspConstraints,
            PrimitiveType.RELEASE: ReleaseConstraints,
            PrimitiveType.PLACE: PlaceConstraints,
            PrimitiveType.ALIGN: AlignConstraints,
            PrimitiveType.INSERT: InsertConstraints,
            PrimitiveType.WAIT: WaitConstraints,
            PrimitiveType.CHECK: CheckConstraints,
        }
        return mapping[self.type](**self.constraints)


class VerificationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps_total: int = Field(..., ge=0)
    steps_executed: int = Field(..., ge=0)
    goal_reached: bool
    collision_detected: bool


class VerificationError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error_id: str
    step_id: str
    type: str
    severity: ErrorSeverity
    message: str
    entities: List[str] = Field(default_factory=list)
    suggested_repairs: List[str] = Field(default_factory=list)


class VerificationMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_clearance: Optional[float] = Field(default=None, ge=0.0)
    max_joint_velocity_ratio: Optional[float] = Field(default=None, ge=0.0)
    estimated_execution_time_sec: Optional[float] = Field(default=None, ge=0.0)


@dataclass
class StepTrace:
    step_id: str
    step_type: str
    before_state: Dict[str, Any]
    after_state: Dict[str, Any]
    success: bool
    error: Optional[Dict[str, Any]] = None


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    simulator: str
    status: Literal["passed", "failed"]
    summary: VerificationSummary
    errors: List[VerificationError] = Field(default_factory=list)
    metrics: Optional[VerificationMetrics] = None
    traces: List[StepTrace] = Field(default_factory=list)


class RepairPatchOp(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_step_id: str
    op: RepairOp
    path: str
    value: Any = None


class RepairPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    based_on_error: str
    patches: List[RepairPatchOp] = Field(default_factory=list)
    reason: str


class GenericCobotIR(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ir_version: str = "0.1"
    task_id: str
    created_by: str = "nl_parser"
    created_at: datetime
    task_spec: TaskSpec
    robot_profile: RobotProfile
    world_binding: WorldBinding
    action_plan: List[ActionStep] = Field(default_factory=list)
    verification_policy: VerificationPolicy
    repair_state: RepairState = Field(default_factory=RepairState)

    @field_validator("action_plan")
    @classmethod
    def validate_unique_step_ids(cls, steps: List[ActionStep]) -> List[ActionStep]:
        ids = [s.step_id for s in steps]
        if len(ids) != len(set(ids)):
            raise ValueError("step_id values must be unique")
        return steps
