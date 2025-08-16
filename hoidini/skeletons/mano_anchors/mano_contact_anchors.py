import torch
import json
import os
from functools import lru_cache

from hoidini.general_utils import SRC_DIR


@lru_cache()
def get_contact_anchors_info():
    """
    R2x meaning that the indices belong to x's mesh
    """
    with open(
        os.path.join(SRC_DIR, "skeletons/mano_anchors/mano_contact_anchors_data.json"),
        "r",
    ) as f:
        anchors_data = json.load(f)
    anchor_inds_R2hands = torch.tensor(anchors_data["anchor_indices"])
    closest_anchor_per_vertex_R2anchors = torch.tensor(
        anchors_data["closest_anchor_per_vertex"]
    )

    closest_anchor_per_vertex_R2hands = anchor_inds_R2hands[
        closest_anchor_per_vertex_R2anchors
    ]
    return (
        anchor_inds_R2hands,
        closest_anchor_per_vertex_R2anchors,
        closest_anchor_per_vertex_R2hands,
    )
