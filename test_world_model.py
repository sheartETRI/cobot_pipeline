import json
from pathlib import Path
from world_model import WorldModel

for path in [
    "samples/sample_pick_place.world.json",
    "samples/sample_insert.world.json",
]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    wm = WorldModel.model_validate(data)
    print("=" * 60)
    print("file:", path)
    print("objects:", list(wm.objects.keys()))
    print("features:", list(wm.features.keys()))
    print("frames:", list(wm.frames.keys()))
    print("tcp:", wm.robot_state.tcp_pose.position)
