from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


# =========================
# Basic Pose / Transform Models
# =========================

class Pose(BaseModel):
    """
    Pose in a named reference frame.
    position: [x, y, z]
    orientation: quaternion [qx, qy, qz, qw]
    """
    model_config = ConfigDict(extra="forbid")

    frame: str = "world"
    position: List[float] = Field(..., min_length=3, max_length=3)
    orientation: List[float] = Field(..., min_length=4, max_length=4)

    @field_validator("position")
    @classmethod
    def validate_position(cls, v: List[float]) -> List[float]:
        if len(v) != 3:
            raise ValueError("position must have exactly 3 elements")
        return v

    @field_validator("orientation")
    @classmethod
    def validate_orientation(cls, v: List[float]) -> List[float]:
        if len(v) != 4:
            raise ValueError("orientation must have exactly 4 elements")
        return v


# =========================
# Geometry Models
# =========================

class BoxGeometry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["box"] = "box"
    size: List[float] = Field(..., min_length=3, max_length=3)  # x, y, z

    @field_validator("size")
    @classmethod
    def validate_size(cls, v: List[float]) -> List[float]:
        if len(v) != 3:
            raise ValueError("box size must have exactly 3 elements")
        if any(x <= 0 for x in v):
            raise ValueError("box size values must be positive")
        return v


class CylinderGeometry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["cylinder"] = "cylinder"
    radius: float = Field(..., gt=0.0)
    height: float = Field(..., gt=0.0)


class PlaneGeometry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["plane"] = "plane"
    size: List[float] = Field(..., min_length=2, max_length=2)  # x, y

    @field_validator("size")
    @classmethod
    def validate_size(cls, v: List[float]) -> List[float]:
        if len(v) != 2:
            raise ValueError("plane size must have exactly 2 elements")
        if any(x <= 0 for x in v):
            raise ValueError("plane size values must be positive")
        return v


Geometry = Union[BoxGeometry, CylinderGeometry, PlaneGeometry]


# =========================
# Object / Feature Models
# =========================

class ObjectModel(BaseModel):
    """
    Physical object in the world.
    """
    model_config = ConfigDict(extra="forbid")

    object_id: str
    object_type: str
    pose: Pose
    geometry: Geometry
    movable: bool = True
    graspable: bool = True
    collision_enabled: bool = True
    metadata: Dict[str, str] = Field(default_factory=dict)


class FeatureModel(BaseModel):
    """
    Feature belonging to a parent object.
    local_pose is expressed in the parent object frame.
    """
    model_config = ConfigDict(extra="forbid")

    feature_id: str
    parent_object: str
    feature_type: Literal["surface", "slot", "pocket", "hole", "grasp_region"]
    local_pose: Pose
    size_hint: Optional[List[float]] = None
    axis: Optional[List[float]] = None
    depth: Optional[float] = Field(default=None, gt=0.0)
    width: Optional[float] = Field(default=None, gt=0.0)
    metadata: Dict[str, str] = Field(default_factory=dict)

    @field_validator("axis")
    @classmethod
    def validate_axis(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is None:
            return v
        if len(v) != 3:
            raise ValueError("axis must have exactly 3 elements")
        return v


class FrameModel(BaseModel):
    """
    Explicit frame registry entry.
    pose is expressed in the parent frame named by pose.frame.
    """
    model_config = ConfigDict(extra="forbid")

    frame_id: str
    registry_path: str
    pose: Pose
    metadata: Dict[str, str] = Field(default_factory=dict)


# =========================
# Robot State Model
# =========================

class RobotState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_frame: str = "world"
    tcp_pose: Pose
    joint_positions: List[float] = Field(default_factory=list)
    attached_object: Optional[str] = None


# =========================
# Root World Model
# =========================

class WorldModel(BaseModel):
    """
    Minimal geometry-aware world model for simulator integration.
    """
    model_config = ConfigDict(extra="forbid")

    world_frame: str = "world"
    objects: Dict[str, ObjectModel] = Field(default_factory=dict)
    features: Dict[str, FeatureModel] = Field(default_factory=dict)
    frames: Dict[str, FrameModel] = Field(default_factory=dict)
    robot_state: RobotState

    @field_validator("features")
    @classmethod
    def validate_feature_parents(cls, features: Dict[str, FeatureModel], info):
        objects = info.data.get("objects", {})
        for feature_id, feature in features.items():
            if feature.parent_object not in objects:
                raise ValueError(
                    f"feature '{feature_id}' refers to unknown parent_object "
                    f"'{feature.parent_object}'"
                )
        return features

    @field_validator("frames")
    @classmethod
    def validate_frames(cls, frames: Dict[str, FrameModel], info):
        world_frame = info.data.get("world_frame", "world")
        objects = info.data.get("objects", {})
        features = info.data.get("features", {})
        known_refs = {world_frame, *objects.keys(), *features.keys()}

        for frame_id, frame in frames.items():
            if frame.frame_id != frame_id:
                raise ValueError(
                    f"frame key '{frame_id}' must match frame_id '{frame.frame_id}'"
                )
            if frame.pose.frame not in known_refs and frame.pose.frame not in frames:
                raise ValueError(
                    f"frame '{frame_id}' refers to unknown parent frame '{frame.pose.frame}'"
                )
        return frames

    def get_object(self, object_id: str) -> ObjectModel:
        if object_id not in self.objects:
            raise KeyError(f"unknown object: {object_id}")
        return self.objects[object_id]

    def get_feature(self, feature_id: str) -> FeatureModel:
        if feature_id not in self.features:
            raise KeyError(f"unknown feature: {feature_id}")
        return self.features[feature_id]

    def has_object(self, object_id: str) -> bool:
        return object_id in self.objects

    def has_feature(self, feature_id: str) -> bool:
        return feature_id in self.features

    def get_frame(self, frame_id: str) -> FrameModel:
        if frame_id not in self.frames:
            raise KeyError(f"unknown frame: {frame_id}")
        return self.frames[frame_id]

    def has_frame(self, frame_id: str) -> bool:
        return frame_id in self.frames
