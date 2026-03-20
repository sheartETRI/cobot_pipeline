#!/usr/bin/env python3
# scripts/generate_and_run_rule.py
"""
간단 규칙 기반 자연어 -> world.json + ir.json 생성 후 run_demo로 실행하는 스크립트.
사용 예:
  python scripts/generate_and_run_rule.py \
    --nl "pick red block at (0.3,0.1,0.02) and place it at (0.55,0.12,0.01)" \
    --out-prefix sample_pick_place --pybullet-gui

옵션:
  --nl             자연어 입력(따옴표로 묶어서 전달). 없으면 인터랙티브 입력.
  --out-prefix     생성할 파일 접두어 (기본: sample_pick_place)
  --no-run         생성만 하고 시뮬레이션은 실행하지 않음
  --sim-backend    시뮬레이터 backend (기본: pybullet)
  --pybullet-gui   pybullet GUI 사용 여부 (기본: False)
"""
from __future__ import annotations

import re
import json
import uuid
import sys
import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# pydantic 모델(레포리의 모듈 경로 기준으로 실행)
from ir_models import GenericCobotIR
from world_model import WorldModel
from datetime import datetime

SAMPLES_DIR = ROOT_DIR / "samples"
SAMPLES_DIR.mkdir(exist_ok=True)

def resolve_prefix(prefix: str, force: bool) -> str:
    ipath = SAMPLES_DIR / f"{prefix}.json"
    wpath = SAMPLES_DIR / f"{prefix}.world.json"
    if force:
        return prefix
    if not ipath.exists() and not wpath.exists():
        return prefix
    # 둘 중 하나라도 존재하면 타임스탬프를 붙인 새 접두어 사용
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    new_prefix = f"{prefix}-{ts}"
    print(f"[INFO] '{prefix}' 파일이 이미 존재합니다. 덮어쓰지 않고 새 파일을 만듭니다: '{new_prefix}'")
    return new_prefix

def parse_positions(nl: str) -> list[Tuple[float, float, float]]:
    """
    자연어에서 (x,y,z) 또는 x,y,z 형태의 좌표들을 찾음.
    예: "(0.3, 0.1, 0.02)" 또는 "0.3,0.1,0.02"
    """
    coords = re.findall(r"\(?\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)?", nl)
    return [(float(x), float(y), float(z)) for x, y, z in coords]

def build_world(block_pos: Tuple[float,float,float], target_pos: Tuple[float,float,float]) -> dict:
    """샘플 기반 간단한 world JSON 생성"""
    world = {
        "world_frame": "world",
        "objects": {
            "block_red": {
                "object_id": "block_red",
                "object_type": "block",
                "pose": {
                    "frame": "world",
                    "position": list(block_pos),
                    "orientation": [0.0, 0.0, 0.0, 1.0]
                },
                "geometry": {"type": "box", "size": [0.04, 0.04, 0.04]},
                "movable": True,
                "graspable": True,
                "collision_enabled": True,
                "metadata": {"color": "red", "registry_id": "obj_block_red_01"}
            },
            "target_zone": {
                "object_id": "target_zone",
                "object_type": "fixture",
                "pose": {
                    "frame": "world",
                    "position": list(target_pos),
                    "orientation": [0.0, 0.0, 0.0, 1.0]
                },
                "geometry": {"type": "box", "size": [0.2, 0.15, 0.02]},
                "movable": False,
                "graspable": False,
                "collision_enabled": True,
                "metadata": {"role": "placement_fixture", "registry_id": "obj_target_zone_01"}
            }
        },
        "features": {
            "block_red_top_surface": {
                "feature_id": "block_red_top_surface",
                "parent_object": "block_red",
                "feature_type": "surface",
                "local_pose": {
                    "frame": "block_red",
                    "position": [0.0, 0.0, 0.02],
                    "orientation": [0.0, 0.0, 0.0, 1.0]
                },
                "size_hint": [0.04, 0.04],
                "metadata": {"usage": "grasp_reference_surface"}
            },
            "target_surface": {
                "feature_id": "target_surface",
                "parent_object": "target_zone",
                "feature_type": "surface",
                "local_pose": {
                    "frame": "target_zone",
                    "position": [0.0, 0.0, 0.01],
                    "orientation": [0.0, 0.0, 0.0, 1.0]
                },
                "size_hint": [0.18, 0.12],
                "metadata": {"usage": "flat_placement_surface"}
            }
        },
        "frames": {
            "block_red_frame": {
                "frame_id": "block_red_frame",
                "registry_path": "obj_block_red_01/frame",
                "pose": {
                    "frame": "world",
                    "position": list(block_pos),
                    "orientation": [0.0,0.0,0.0,1.0]
                },
                "metadata": {"source":"object_pose"}
            },
            "target_zone_frame": {
                "frame_id": "target_zone_frame",
                "registry_path": "obj_target_zone_01/frame",
                "pose": {
                    "frame": "world",
                    "position": list(target_pos),
                    "orientation": [0.0,0.0,0.0,1.0]
                },
                "metadata": {"source":"object_pose"}
            },
            "home_frame": {
                "frame_id": "home_frame",
                "registry_path": "robot/home_pose",
                "pose": {
                    "frame": "world",
                    "position": [0.4, 0.0, 0.3],
                    "orientation": [0.0,0.0,0.0,1.0]
                },
                "metadata": {"source":"robot_home"}
            }
        },
        "robot_state": {
            "base_frame": "world",
            "tcp_pose": {
                "frame": "world",
                "position": [0.4, 0.0, 0.3],
                "orientation": [0.0, 0.0, 0.0, 1.0]
            },
            "joint_positions": [0.0, -1.2, 1.5, 0.0, 1.2, 0.0],
            "attached_object": None
        }
    }
    return world

def build_ir(nl: str, task_suffix: str = "") -> dict:
    """단순한 pick-and-place action_plan을 가지는 IR 생성"""
    now = datetime.now(timezone.utc).isoformat()
    task_id = f"task_pick_place_{uuid.uuid4().hex[:8]}" if not task_suffix else f"task_{task_suffix}_{uuid.uuid4().hex[:6]}"
    ir = {
        "ir_version": "0.1",
        "task_id": task_id,
        "created_by": "nl_generator",
        "created_at": now,
        "task_spec": {
            "goal": "pick_and_place",
            "command_text": nl,
            "priority": "normal",
            "success_condition": ["object_at_destination:block_red:target_surface"],
            "assumptions": []
        },
        "robot_profile": {
            "robot_type": "generic_cobot",
            "arm_dof": 6,
            "has_gripper": True,
            "tool_frame": "gripper_tcp",
            "base_frame": "world",
            "motion_limits_profile": "default_cobot"
        },
        "world_binding": {
            "scene_id": "sample_scene",
            "objects": { "block_red": "obj_block_red_01", "target_zone": "obj_target_zone_01" },
            "frames": { "block_red_frame": "obj_block_red_01/frame", "target_zone_frame": "obj_target_zone_01/frame", "home_frame":"robot/home_pose" },
            "regions": {},
            "features": {
                "block_red_top_surface": {
                    "parent_object": "block_red",
                    "feature_type": "surface",
                    "frame": "block_red_frame",
                    "description":"top surface"
                },
                "target_surface": {
                    "parent_object": "target_zone",
                    "feature_type": "surface",
                    "frame": "target_zone_frame",
                    "description":"placement surface"
                }
            }
        },
        "action_plan": [
            {"step_id":"s1","type":"find_object","inputs":{"object":"block_red"}},
            {"step_id":"s2","type":"approach","inputs":{"target_object":"block_red","target_feature":"block_red_top_surface","approach_pose":{"ref":"block_red_frame","offset":[0.0,0.0,0.08],"orientation_policy":"align_with_object_top"}}},
            {"step_id":"s3","type":"grasp","inputs":{"target_object":"block_red","target_feature":"block_red_top_surface","grasp_mode":"pinch"}},
            {"step_id":"s4","type":"retreat","inputs":{"direction":"tool_z_negative","distance":0.08}},
            {"step_id":"s5","type":"move_linear","inputs":{"target_pose":{"ref":"target_zone_frame","offset":[0.0,0.0,0.1],"orientation_policy":"keep_current"}}},
            {"step_id":"s6","type":"place","inputs":{"target_object":"block_red","target_feature":"target_surface","destination_pose":{"ref":"target_zone_frame","offset":[0.0,0.0,0.0],"orientation_policy":"align_to_target_surface"}}},
            {"step_id":"s7","type":"release","inputs":{"target_object":"block_red"}},
            {"step_id":"s8","type":"retreat","inputs":{"direction":"tool_z_negative","distance":0.1}}
        ],
        "verification_policy":{"collision_check":True,"ik_check":True,"joint_limit_check":True,"velocity_limit_check":True,"force_limit_check":False,"max_retry":3,"acceptance_rules":[]},
        "repair_state": {"retry_count":0,"last_error":None,"repair_history":[]}
    }
    return ir

def save_and_validate(world: dict, ir: dict, prefix: str) -> tuple[str,str]:
    """Pydantic으로 검증 후 files 저장(덮어쓰기)"""
    wpath = SAMPLES_DIR / f"{prefix}.world.json"
    ipath = SAMPLES_DIR / f"{prefix}.json"
    # 검증(ValidationError 발생 시 예외 전파)
    WorldModel.model_validate(world)
    GenericCobotIR.model_validate(ir)
    wpath.write_text(json.dumps(world, indent=2, ensure_ascii=False), encoding="utf-8")
    ipath.write_text(json.dumps(ir, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(ipath), str(wpath)

def run_sim(ir_path: str, sim_backend: str = "pybullet", gui: bool = False, step_wait: bool = False) -> None:
    cmd = [sys.executable, str(ROOT_DIR / "run_demo.py"), "--sample", ir_path, "--sim-backend", sim_backend]
    if gui:
        cmd.append("--pybullet-gui")
    if step_wait:
        cmd.append("--pybullet-step-wait")
    subprocess.run(cmd, check=True, cwd=ROOT_DIR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nl", type=str, default=None, help="자연어 입력 (예: 'pick red block at (0.3,0.1,0.02) and place it at (0.55,0.12,0.01)')")
    parser.add_argument("--out-prefix", type=str, default="sample_pick_place", help="저장할 파일 접두어")
    parser.add_argument("--no-run", action="store_true", help="생성만 하고 시뮬레이션은 실행하지 않음")
    parser.add_argument("--sim-backend", type=str, default="pybullet", choices=["pybullet","mock"], help="시뮬레이터 백엔드")
    parser.add_argument("--pybullet-gui", action="store_true", help="pybullet GUI 사용")
    parser.add_argument("--force", action="store_true", help="기존 파일이 있어도 강제로 덮어씀")
    parser.add_argument("--step-by-step", action="store_true", help="PyBullet GUI에서 각 스텝마다 Enter를 눌러 진행")
    args = parser.parse_args()

    if args.nl:
        nl_text = args.nl
    else:
        nl_text = input("자연어 입력 (예: pick red block at (0.3,0.1,0.02) and place it at (0.55,0.12,0.01)):\n")

    coords = parse_positions(nl_text)
    if len(coords) >= 2:
        block_pos, target_pos = coords[0], coords[1]
    else:
        print("좌표가 충분히 추출되지 않아 기본값 사용합니다.")
        block_pos = (0.3,0.1,0.02)
        target_pos = (0.55,0.12,0.01)

    world = build_world(block_pos, target_pos)
    ir = build_ir(nl_text)
    prefix = args.out_prefix

    try:
        # 기존: prefix = args.out_prefix
        prefix = resolve_prefix(args.out_prefix, force=args.force)
        ir_path, world_path = save_and_validate(world, ir, prefix)
        print(f"생성 및 검증 완료: {ir_path}, {world_path}")
    except Exception as e:
        print("검증 실패:", e)
        raise

    if not args.no_run:
        run_sim(ir_path, sim_backend=args.sim_backend, gui=args.pybullet_gui, step_wait=args.step_by_step)

if __name__ == "__main__":
    main()
