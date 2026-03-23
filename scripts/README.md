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

## `make_world.py`

宣言型 CLI로 `WorldModel` JSON을 바로 만든다.

주요 옵션:

- `--yaml <path>`
- `--block <name>@x,y,z`
- `--fixture <name>@x,y,z[@sx,sy,sz]`
- `--tcp x,y,z`
- `--out <path>`

예제:

```bash
python scripts/make_world.py \
  --block red@0.30,0.10,0.02 \
  --block blue@0.30,-0.10,0.02 \
  --block green@0.45,0.10,0.02 \
  --fixture target_zone@0.55,0.12,0.01@0.20,0.15,0.02 \
  --out samples/world/demo_multi_blocks.world.json
```

동작:

- `red`는 `block_red`로 정규화된다.
- 각 block에는 `<object>_top_surface`와 `<object>_frame`이 자동 생성된다.
- `target` 또는 `target_zone` fixture에는 `target_surface`가 자동 생성된다.

YAML 예제:

```yaml
blocks:
  - name: red
    position: [0.30, 0.10, 0.02]
  - name: blue
    position: [0.30, -0.10, 0.02]
fixtures:
  - name: target
    position: [0.55, 0.12, 0.01]
    size: [0.20, 0.15, 0.02]
tcp: [0.4, 0.0, 0.3]
```

```bash
python scripts/make_world.py --yaml samples/world/demo_multi_blocks.yaml --out samples/world/demo_multi_blocks.world.json
```

## `interactive_session.py`

대화형 세션으로 자연어 명령을 순차 실행한다. 각 명령이 끝나면 마지막 world 상태를 다음 명령이 이어받는다.

주요 옵션:

- `--existing-world <path>`
- `--out-prefix <prefix>`
- `--save-session`
- `--save-final-world <path>`
- `--sim-backend mock|pybullet`
- `--pybullet-gui`
- `--step-by-step`
- `--pybullet-motion-delay <seconds>`
- `--pybullet-step-pause <seconds>`

예제:

```bash
python scripts/interactive_session.py \
  --existing-world samples/world/demo_multi_blocks.world.json \
  --out-prefix session_demo \
  --sim-backend pybullet \
  --pybullet-gui
```

세션 입력 예시:

```text
pick the red block and place it on the target surface
pick the blue block and place it on the red block surface
pick the green block and place it on the blue block surface
exit
```

한 줄에 `;`로 구분된 연속 명령도 가능하다.

```text
pick the red block and place it on the target surface; pick the blue block and place it on the red block surface
```

종료 동작:

- 기본값: 세션 중 생성한 `samples/ir/<prefix>-NNN.json`과 `samples/world/<prefix>_latest.world.json`을 종료 시 삭제
- `--save-session`: 세션 산출물을 유지
- `--save-final-world <path>`: 마지막 world만 지정 경로에 저장
