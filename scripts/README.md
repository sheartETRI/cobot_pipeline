# Scripts

## `generate_and_run_rule.py`

자연어 명령에서 pick/place IR을 만들고 `run_demo.py`를 호출한다.

주요 옵션:

- `--existing-world <path>`: 기존 `WorldModel` JSON 사용
- `--world-from-pybullet`: 현재 PyBullet 세션을 export해서 사용
- `--pybullet-gui`: PyBullet GUI 사용
- `--step-by-step`: GUI에서 각 primitive step마다 Enter 입력 후 진행

예제 1: 기존 world 사용

```bash
python scripts/generate_and_run_rule.py \
  --nl "pick red block at (0.3,0.1,0.02) and place it at (0.55,0.12,0.01)" \
  --existing-world samples/world/pybullet_exported_world.json \
  --out-prefix sample_pick_place \
  --pybullet-gui
```

예제 2: 현재 PyBullet 세션에서 export 후 사용

```bash
python -c "from scripts.pybullet_world_utils import connect, create_box, create_fixture_box, add_feature; connect(gui=True); create_box('obj_block_red_01', (0.3,0.1,0.02)); create_fixture_box('obj_target_zone_01', (0.55,0.12,0.01)); add_feature('block_red_top_surface', 'obj_block_red_01', (0,0,0.02), size_hint=[0.04,0.04]); add_feature('target_surface', 'obj_target_zone_01', (0,0,0.01), size_hint=[0.2,0.15]); input('Press Enter to keep the session alive...')"
python scripts/generate_and_run_rule.py \
  --nl "pick red block at (0.3,0.1,0.02) and place it at (0.55,0.12,0.01)" \
  --world-from-pybullet \
  --out-prefix sample_pick_place \
  --pybullet-gui
```

제약:

- `--world-from-pybullet`는 동일 프로세스의 active PyBullet 연결 또는 shared-memory 세션이 있어야 한다.
- 기존 world에 필수 alias `block_red`, `target_zone`가 없으면 자동 추정하지 않고 에러를 낸다.
- 누락된 frame/feature는 자동 보강되며, 생성된 항목과 payload는 로그로 출력된다.
- export 후 연결 종료 여부를 제어하는 CLI 옵션은 아직 없다.

## `pybullet_world_utils.py`

PyBullet 장면을 만들고 `WorldModel` JSON으로 export하는 유틸.

제공 함수:

- `connect(gui: bool = True) -> int`
- `create_box(...) -> int`
- `create_fixture_box(...) -> int`
- `add_feature(...) -> None`
- `export_world_model(path: str = "samples/world/pybullet_exported_world.json", world_frame: str = "world") -> str`
