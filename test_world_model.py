import json
from pathlib import Path

from sample_paths import world_sample_path
from world_model import WorldModel

for path in [
    str(world_sample_path("sample_pick_place")),
    str(world_sample_path("sample_insert")),
]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    wm = WorldModel.model_validate(data)
    print("=" * 60)
    print("file:", path)
    print("objects:", list(wm.objects.keys()))
    print("features:", list(wm.features.keys()))
    print("frames:", list(wm.frames.keys()))
    print("tcp:", wm.robot_state.tcp_pose.position)
