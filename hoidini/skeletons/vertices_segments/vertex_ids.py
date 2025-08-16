import os
import json
from typing import Dict, List


def get_segments_vertex_ids_dict(model_name) -> Dict[str, List[int]]:
    assert model_name in ["smpl", "smplx"]
    file_path = os.path.join(
        os.path.dirname(__file__), f"{model_name}_vert_segmentation.json"
    )
    with open(file_path, "r") as f:
        part_segm = json.load(f)
    return part_segm


if __name__ == "__main__":
    d = get_segments_vertex_ids_dict("smpl")
    print(d)
