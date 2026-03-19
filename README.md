# Cobot Pipeline Feature Pick/Place World Load Patch

이 저장소는 코봇 작업 IR(`GenericCobotIR`)에 대해 간단한 실행/검증을 수행하고, 별도 `world model`을 로드해 객체, feature, frame 정보를 함께 검증하는 예제 코드입니다.

최근 작업에서 추가된 핵심은 다음 두 가지입니다.

- `world_model.py` 기반의 명시적 월드 모델
- `frames` 레지스트리 추가
  - `home_frame`
  - `*_frame` 형태의 object / feature / waypoint frame

## 주요 파일

- `world_model.py`
  - 월드 모델 스키마
  - `objects`, `features`, `frames`, `robot_state` 정의
- `ir_models.py`
  - IR 스키마 정의
- `executor.py`
  - mock simulator 기반 IR 실행기
- `run_demo.py`
  - 샘플 IR / world 파일 로드 및 실행
- `test_world_model.py`
  - 샘플 world JSON 로딩 확인용 간단 테스트 스크립트
- `tests/test_world_integration.py`
  - world model 연동 테스트
- `samples/`
  - 샘플 IR 및 world JSON

## world model 구조

`WorldModel`은 아래 정보를 가집니다.

- `world_frame`
- `objects`
  - 월드에 존재하는 물체의 pose / geometry
- `features`
  - object에 종속된 surface / slot / pocket / hole / grasp_region
- `frames`
  - 명시적 frame registry
  - 예: `block_red_frame`, `target_zone_frame`, `pin_frame`, `hole_frame`, `pre_insert_frame`, `home_frame`
- `robot_state`
  - `base_frame`, `tcp_pose`, `joint_positions`, `attached_object`

`frames`는 IR의 `world_binding.frames`와 직접 대응시키기 위한 용도입니다.
기존처럼 object / feature만으로 간접적으로 frame을 추론하지 않고, 월드 파일 안에서 frame을 명시적으로 선언합니다.

## 샘플 데이터

현재 샘플 world 파일에는 frame registry가 포함되어 있습니다.

- `samples/sample_pick_place.world.json`
  - `block_red_frame`
  - `target_zone_frame`
  - `home_frame`
- `samples/sample_insert.world.json`
  - `pin_frame`
  - `hole_frame`
  - `pre_insert_frame`

## 실행 환경

권장 환경:

- Python 3.10+
- `pydantic`

필요 시 기본 패키지 설치:

```bash
pip install pydantic pytest
```

### 실행 방법

# 1. world model 로딩 확인

샘플 world JSON이 정상적으로 파싱되는지 확인합니다.

python test_world_model.py
확인 내용:

샘플 world 파일 로딩
object / feature / frame 목록 출력
robot tcp 정보 출력

# 2. 샘플 데모 실행

IR와 world model을 함께 로드해 mock executor를 실행합니다.

python run_demo.py
확인 내용:

샘플 IR 로딩
대응되는 \*.world.json 자동 탐색
IR와 world model consistency check
step-by-step 실행 결과 출력
verification result JSON 출력

# 3. 통합 테스트 실행

pytest 기반 테스트를 실행합니다.

전체 테스트:

pytest
world integration 테스트만 실행:

pytest tests/test_world_integration.py
현재 검증 포인트
현재 코드 기준으로 다음 항목을 확인할 수 있습니다.

world JSON이 WorldModel 스키마에 맞게 로딩되는지
feature가 유효한 parent object를 참조하는지
frame registry의 key와 frame_id가 일치하는지
frame의 부모 참조(pose.frame)가 유효한 world / object / feature / frame 인지
IR의 world_binding.frames가 world.frames와 일치하는지
샘플 pick/place 및 insert 시나리오가 mock executor에서 실행 가능한지

# 빠른 시작

가장 빠른 확인 순서는 아래와 같습니다.

python test_world_model.py
python run_demo.py
pytest tests/test_world_integration.py

참고
run_demo.py는 world 파일이 없는 샘플에 대해서는 IR만으로도 실행합니다.
world consistency check는 현재 world.frames를 기준으로 frame alias를 검증합니다.
frame registry를 확장할 때는 samples/\*.world.json에도 대응 frame을 같이 추가하는 것이 좋습니다.
