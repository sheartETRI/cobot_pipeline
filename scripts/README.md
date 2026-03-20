# Scripts

## `generate_and_run_rule.py`

자연어 명령에서 pick/place IR을 만들고, 새 월드를 생성하거나 기존 월드를 사용해 `run_demo.py`를 실행합니다.

### 예제 1: 이미 저장된 월드 사용

```bash
python scripts/generate_and_run_rule.py \
  --nl "pick red block at (0.3,0.1,0.02) and place it at (0.55,0.12,0.01)" \
  --existing-world samples/pybullet_exported_world.json \
  --out-prefix sample_pick_place \
  --pybullet-gui
```

### 예제 2: 현재 PyBullet 세션에서 월드 추출

```bash
python -c "from scripts.pybullet_world_utils import connect, create_box, create_fixture_box, add_feature; connect(gui=True); create_box('obj_block_red_01', (0.3,0.1,0.02)); create_fixture_box('obj_target_zone_01', (0.55,0.12,0.01)); add_feature('block_red_top_surface', 'obj_block_red_01', (0,0,0.02)); add_feature('target_surface', 'obj_target_zone_01', (0,0,0.01)); input('Press Enter to keep the session alive...')"
python scripts/generate_and_run_rule.py \
  --nl "pick red block at (0.3,0.1,0.02) and place it at (0.55,0.12,0.01)" \
  --world-from-pybullet \
  --out-prefix sample_pick_place \
  --pybullet-gui
```

주의:

- `--world-from-pybullet`는 현재 프로세스의 활성 PyBullet 연결 또는 shared-memory 세션이 있어야 합니다.
- 같은 터미널/프로세스에서 scene을 만들고 바로 `generate_and_run_rule.main(...)`을 호출하는 테스트/자동화 흐름도 지원합니다.
- `--step-by-step`을 함께 주면 PyBullet GUI에서 각 primitive step마다 Enter를 눌러 다음 스텝으로 진행합니다.

## `pybullet_world_utils.py`

PyBullet 장면을 구성하고 `WorldModel` 형식 JSON으로 내보내는 유틸입니다.

제공 함수:

- `connect(gui: bool = True) -> int`
- `create_box(...) -> int`
- `create_fixture_box(...) -> int`
- `add_feature(...) -> None`
- `export_world_model(path: str = "pybullet_exported_world.json", world_frame: str = "world") -> str`
