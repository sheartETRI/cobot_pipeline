# Cobot Pipeline

이 저장소는 코봇 작업 IR(`GenericCobotIR`)과 월드 모델(`WorldModel`)을 생성, 검증, 실행하는 예제 코드다. 현재는 기본 샘플 월드 생성뿐 아니라 기존 월드 재사용, PyBullet 세션에서의 월드 export, step-by-step 실행까지 지원한다.

## 주요 파일

- `ir_models.py`: IR 스키마
- `world_model.py`: WorldModel 스키마
- `executor.py`: mock / PyBullet 실행기
- `run_demo.py`: IR + world 로드, 일관성 검사, 실행
- `scripts/generate_and_run_rule.py`: 자연어로 IR 생성 및 실행
- `scripts/pybullet_world_utils.py`: PyBullet 장면 구성 및 WorldModel export
- `tests/test_world_integration.py`: world/IR 통합 테스트
- `tests/test_pybullet_world_export.py`: PyBullet export 및 existing-world 연동 테스트

## 지원 흐름

### 1. 기본 world 생성 후 IR 생성/실행

```bash
python scripts/generate_and_run_rule.py \
  --nl "pick red block at (0.3,0.1,0.02) and place it at (0.55,0.12,0.01)" \
  --out-prefix sample_pick_place \
  --sim-backend pybullet \
  --pybullet-gui
```

### 2. 기존 WorldModel JSON 사용

```bash
python scripts/generate_and_run_rule.py \
  --nl "pick red block at (0.3,0.1,0.02) and place it at (0.55,0.12,0.01)" \
  --existing-world samples/pybullet_exported_world.json \
  --out-prefix sample_pick_place \
  --sim-backend pybullet \
  --pybullet-gui
```

이 경우 `build_world()`는 건너뛰고 기존 world의 object/frame/feature를 기준으로 `world_binding`을 채운다.

필수 alias:

- `block_red`
- `target_zone`

권장 frame / feature:

- `block_red_frame`
- `target_zone_frame`
- `home_frame`
- `block_red_top_surface`
- `target_surface`

누락된 frame / feature는 경고 후 자동 보강한다. 자동 추정으로 다른 object alias를 매칭하지는 않는다. 즉 `block_red` 또는 `target_zone` alias가 없으면 에러로 종료한다.

### 3. 현재 PyBullet 세션에서 world export 후 사용

같은 프로세스에서 직접 호출하는 방식:

```python
from scripts.pybullet_world_utils import connect, create_box, create_fixture_box, add_feature
from scripts.generate_and_run_rule import main

connect(gui=False)
create_box("obj_block_red_01", (0.3, 0.1, 0.02))
create_fixture_box("obj_target_zone_01", (0.55, 0.12, 0.01))
add_feature("block_red_top_surface", "obj_block_red_01", (0.0, 0.0, 0.02), size_hint=[0.04, 0.04])
add_feature("target_surface", "obj_target_zone_01", (0.0, 0.0, 0.01), size_hint=[0.2, 0.15])

main([
    "--nl", "pick red block at (0.3,0.1,0.02) and place it at (0.55,0.12,0.01)",
    "--world-from-pybullet",
    "--out-prefix", "sample_pick_place",
    "--sim-backend", "mock",
    "--force",
])
```

CLI 예시:

```bash
python -c "from scripts.pybullet_world_utils import connect, create_box, create_fixture_box, add_feature; connect(gui=True); create_box('obj_block_red_01', (0.3,0.1,0.02)); create_fixture_box('obj_target_zone_01', (0.55,0.12,0.01)); add_feature('block_red_top_surface', 'obj_block_red_01', (0,0,0.02), size_hint=[0.04,0.04]); add_feature('target_surface', 'obj_target_zone_01', (0,0,0.01), size_hint=[0.2,0.15]); input('Press Enter to keep the session alive...')"
python scripts/generate_and_run_rule.py \
  --nl "pick red block at (0.3,0.1,0.02) and place it at (0.55,0.12,0.01)" \
  --world-from-pybullet \
  --out-prefix sample_pick_place \
  --sim-backend pybullet \
  --pybullet-gui
```

## `generate_and_run_rule.py` 옵션

- `--nl <text>`: 자연어 명령
- `--out-prefix <name>`: 출력 파일 prefix
- `--no-run`: JSON만 생성
- `--sim-backend {mock,pybullet}`: 실행 백엔드
- `--pybullet-gui`: PyBullet GUI 사용
- `--step-by-step`: PyBullet GUI에서 각 primitive step마다 Enter 입력 후 진행
- `--existing-world <path>`: 기존 `WorldModel` JSON 사용
- `--world-from-pybullet`: 현재 PyBullet 세션에서 world export 후 사용
- `--force`: 기존 출력 파일 덮어쓰기

스크립트는 항상 다음 순서로 검증/실행한다.

1. `WorldModel.model_validate(...)`
2. `GenericCobotIR.model_validate(...)`
3. `run_demo.py --sample <ir> --world <world>`

최종 object / frame / feature 일관성 검사는 `run_demo.py`의 `validate_ir_world_consistency()`가 다시 수행한다.

## `pybullet_world_utils.py`

제공 함수:

- `connect(gui: bool = True) -> int`
- `create_box(...) -> int`
- `create_fixture_box(...) -> int`
- `add_feature(...) -> None`
- `export_world_model(path: str = "pybullet_exported_world.json", world_frame: str = "world") -> str`

export 시 다음을 포함한다.

- object `metadata.registry_id`
- object pose / geometry / movable / graspable / collision_enabled
- feature local pose
- frame registry
- robot_state
- `WorldModel` 검증 후 JSON 저장

registry가 없는 body는 `auto_obj_<body_id>`로 export하며 경고를 남긴다.

## 자연어 좌표 처리

자연어에서 좌표가 추출되면:

- 기본 world 생성 모드에서는 object pose를 해당 좌표로 생성한다.
- 기존 world 사용 모드에서는 object의 실제 pose는 유지한다.
- 대신 target 관련 feature local pose를 조정해 placement 기준점을 반영한다.

## 주의사항 / 현재 제약

### `--world-from-pybullet` 세션 의존성

현재 구현은 동일 프로세스의 active PyBullet 연결 또는 shared-memory 세션이 있어야 한다. 별도 GUI 프로세스에서 생성된 일반 세션을 다른 프로세스가 그대로 읽는 흐름은 기본 지원하지 않는다.

실무적으로는 다음 방식으로 확장 가능하다.

- `p.connect(p.SHARED_MEMORY)` 기반 exporter / consumer 구성
- 별도 exporter 프로세스가 world JSON을 저장하고, generator는 그 JSON만 읽는 구조

### 자동 보강 로직의 보수성

기존 world에 필요한 frame / feature가 없으면 자동 보강한다. 이 동작은 편리하지만 잘못된 보강이 실제 일관성 문제를 가릴 수 있다.

현재는 보강 시 다음 정보를 로그로 남긴다.

- 생성한 항목 이름
- 생성 근거
- 생성 payload 전체

운영 환경에서는 이 로그를 확인해 자동 보강 결과가 기대와 맞는지 검토하는 것이 좋다.

### 객체 alias 불일치 처리 정책

현재는 자동 추정 없이 에러를 낸다. 운영 환경에서 alias 명칭이 자주 다르다면, 향후 명시적 매핑 파일(YAML/JSON)을 받아 처리하는 옵션을 추가하는 것이 적절하다.

### PyBullet 연결 종료 처리

현재는 `--world-from-pybullet` export 후 연결 종료 여부를 CLI에서 선택할 수 없다. GUI 유지/종료 시나리오를 모두 다루려면 향후 `--detach-after-export` 같은 플래그를 추가하는 것이 자연스럽다.

## 테스트

```bash
pytest tests/test_world_integration.py tests/test_pybullet_world_export.py
```

현재 검증 항목:

- sample world/IR 실행
- registry_id 누락 및 mismatch 처리
- collision 검출
- simulator backend fallback
- PyBullet world export 후 `WorldModel` 검증
- `--world-from-pybullet` 경로에서 IR 생성 및 `run_demo.py` 연동
