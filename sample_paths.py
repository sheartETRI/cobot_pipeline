from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT_DIR / "samples"
IR_SAMPLES_DIR = SAMPLES_DIR / "ir"
WORLD_SAMPLES_DIR = SAMPLES_DIR / "world"


def ensure_sample_dirs() -> None:
    SAMPLES_DIR.mkdir(exist_ok=True)
    IR_SAMPLES_DIR.mkdir(exist_ok=True)
    WORLD_SAMPLES_DIR.mkdir(exist_ok=True)


def ir_sample_path(prefix: str) -> Path:
    return IR_SAMPLES_DIR / f"{prefix}.json"


def world_sample_path(prefix: str) -> Path:
    return WORLD_SAMPLES_DIR / f"{prefix}.world.json"


def exported_world_sample_path(prefix: str) -> Path:
    return WORLD_SAMPLES_DIR / f"{prefix}.exported.world.json"


def resolve_world_for_ir(ir_path: str | Path) -> Path | None:
    path = Path(ir_path)
    if path.suffix != ".json" or path.name.endswith(".world.json"):
        return None

    sibling_world = path.with_name(f"{path.stem}.world.json")
    if sibling_world.exists():
        return sibling_world

    if path.parent == IR_SAMPLES_DIR:
        candidate = WORLD_SAMPLES_DIR / f"{path.stem}.world.json"
        if candidate.exists():
            return candidate

    if path.parent == SAMPLES_DIR:
        candidate = WORLD_SAMPLES_DIR / f"{path.stem}.world.json"
        if candidate.exists():
            return candidate

    return None


def discover_ir_samples(samples_dir: str | Path) -> list[str]:
    sample_dir = Path(samples_dir)
    if not sample_dir.exists() or not sample_dir.is_dir():
        return []

    try:
        resolved_sample_dir = sample_dir.resolve()
    except OSError:
        resolved_sample_dir = sample_dir

    patterns = [sample_dir.glob("*.json")]
    if resolved_sample_dir == SAMPLES_DIR.resolve():
        patterns.insert(0, IR_SAMPLES_DIR.glob("*.json"))

    samples: list[str] = []
    for pattern in patterns:
        for path in pattern:
            if path.name.endswith(".world.json"):
                continue
            if path.name.endswith("_repair_patch.json"):
                continue
            if path.name.endswith("_verification.json"):
                continue
            samples.append(str(path))

    return sorted(set(samples))
