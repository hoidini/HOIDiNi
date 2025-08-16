from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
import torch
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from hoidini.amasstools.geometry import axis_angle_to_matrix
from hoidini.closd.diffusion_planner.utils import dist_util
from hoidini.cphoi.cphoi_utils import transform_smpldata
from hoidini.cphoi.samplers.samplers import (
    InferenceJob,
    KitchenSampler,
    surface_loc_to_corners,
)
from hoidini.datasets.grab.grab_utils import (
    get_all_grab_seq_paths,
    get_df_grab_index,
    get_df_prompts,
    get_grab_split_ids,
    get_table_params,
    grab_seq_id_to_seq_path,
    grab_seq_path_to_seq_id,
)
from hoidini.datasets.smpldata import SmplData
from hoidini.object_contact_prediction.cpdm_dno_conds import (
    AboveTableLoss,
    BulbUpLoss,
    KeepObjectStaticLoss,
    Specific6DoFLoss,
    TopVertDistLoss,
)
from scipy.spatial import distance_matrix as cdist
import numpy as np
from hoidini.object_contact_prediction.cpdm_dno_conds import Table
from hoidini.datasets.grab.grab_utils import load_mesh
from hoidini.datasets.grab.grab_utils import grab_seq_id_to_seq_path
from hoidini.datasets.smpldata_preparation import get_extended_smpldata
from objects_fk import ObjectModel


def get_horizontal_table_seq_ids(obj_name):
    """
    Find all grab seq ids where the table is horizontal, use that for the object placement
    """
    df_index = get_df_grab_index()
    all_object_seq_ids = df_index[df_index["object_name"] == obj_name].index.to_list()
    assert len(all_object_seq_ids) > 0, f"No sequences found for object {obj_name}"
    horizontal_table_seq_ids = []
    for seq_id in tqdm(all_object_seq_ids):
        grab_seq_path = grab_seq_id_to_seq_path(seq_id)
        real_table_cornets = get_table_params(grab_seq_path)[2]
        table = Table(real_table_cornets.unsqueeze(0))
        table_normal = table.n[0]
        ez = torch.tensor([0.0, 0.0, 1.0])
        angle = torch.acos(torch.dot(table_normal, ez))
        angle_deg = torch.rad2deg(angle)
        # print(table_normal, angle_deg)
        if angle_deg < 5:
            horizontal_table_seq_ids.append(seq_id)
    return horizontal_table_seq_ids


# sample train grab seq for initial human pose and also object pos

SURFACE_HEIGHTS = {
    "stove": 1.013396,
    "island": 0.985,
    "lader": 0.85,
}


class KitchenSamplerSimple(KitchenSampler):
    def get_prefix_values(self):
        return self.prefix_values

    def get_inference_jobs(self) -> List[InferenceJob]:
        return self.inference_jobs

    def get_phase1_dno_losses(self):
        return self.phase1_dno_losses

    def get_table_corner_locs(self):
        return self.table_corner_locs.unsqueeze(0).reshape(1, 4, 3)

    def __init__(
        self,
        surface_name: str,
        object_name: Optional[str] = None,
        grab_seq_id: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        assert object_name is not None or grab_seq_id is not None
        surface_height = SURFACE_HEIGHTS[surface_name]
        df_index = get_df_grab_index()
        if grab_seq_id is None:
            grab_seq_ids = get_horizontal_table_seq_ids(object_name)
            grab_seq_id = np.random.RandomState(seed).choice(grab_seq_ids)
        else:
            _object_name = df_index.loc[grab_seq_id]["object_name"]
            if object_name is not None:
                assert _object_name == object_name
            object_name = _object_name

        df_prompts = get_df_prompts()
        grab_seq_path = grab_seq_id_to_seq_path(grab_seq_id)
        text = df_prompts.loc[grab_seq_id]["Prompt"]
        smpldata: SmplData = get_extended_smpldata(grab_seq_path)["smpldata"]
        ########################
        # Create a table use the real data table xy and the Kitchen surface height
        ########################
        table_real_corners = get_table_params(grab_seq_path)[2]
        table_real_xy_center = table_real_corners[:, [0, 1]].mean(dim=0)
        table_new_xyz = torch.tensor(
            [table_real_xy_center[0], table_real_xy_center[1], surface_height]
        )
        table_new_corners = surface_loc_to_corners(table_new_xyz, 0.4)

        ########################
        # Place the object on the table
        ########################
        obj_v_template, obj_faces = load_mesh(object_name, 2000)
        obj_pose = smpldata.poses_obj[0]

        obj_fk = ObjectModel(obj_v_template)
        obj_verts_0 = obj_fk(
            global_orient=obj_pose.unsqueeze(0), trans=torch.zeros(1, 3)
        ).vertices
        obj_z = surface_height - obj_verts_0[0, :, 2].min()
        obj_transl = torch.tensor(
            [table_real_xy_center[0], table_real_xy_center[1], obj_z]
        )

        inference_job = InferenceJob(
            grab_seq_path=grab_seq_path,
            text=text,
            object_name=object_name,
            n_frames=115,
            start_frame=0,
        )
        rng_spec_1 = (0, 25)
        rng_spec_2 = (100, 115)
        phase1_dno_losses = {
            "above_table": (
                1.2,
                AboveTableLoss(table_new_corners.unsqueeze(0)).to(dist_util.dev()),
            ),
            "keep_object_static": (0.9, KeepObjectStaticLoss()),
            "specific_6dof_table1": (
                0.5,
                Specific6DoFLoss(
                    obj_transl.unsqueeze(0),
                    obj_pose.unsqueeze(0),
                    target_frames=[torch.arange(rng_spec_1[0], rng_spec_1[1])],
                ).to(dist_util.dev()),
            ),
            "specific_6dof_table2": (
                0.5,
                Specific6DoFLoss(
                    obj_transl.unsqueeze(0),
                    obj_pose.unsqueeze(0),
                    target_frames=[torch.arange(rng_spec_2[0], rng_spec_2[1])],
                ).to(dist_util.dev()),
            ),
        }
        if "bulb" in object_name:
            rng_bulb = (50, 95)
            del phase1_dno_losses["specific_6dof_table2"]
            bulb_loc = obj_transl + torch.tensor(
                [0.0, 0.3, 0.0]
            )  # bulb location is a bit ahead of the table
            bulb_loc[2] = 1.8
            phase1_dno_losses["bulb_up"] = (
                0.6,
                BulbUpLoss(target_frames=[torch.arange(rng_bulb[0], rng_bulb[1])]).to(
                    dist_util.dev()
                ),
            )
            # phase1_dno_losses["bulb_loc"] = (0.6, Specific6DoFLoss(bulb_loc.unsqueeze(0), obj_pose.unsqueeze(0), target_frames=[torch.arange(rng_bulb[0], rng_bulb[1])],
            #                                                        location_only=True).to(dist_util.dev()))
            phase1_dno_losses["top_vert_dist"] = (
                0.7,
                TopVertDistLoss(
                    bulb_loc.unsqueeze(0),
                    target_frames=[torch.arange(rng_bulb[0], rng_bulb[1])],
                ).to(dist_util.dev()),
            )

        self.inference_jobs = [inference_job]
        self.phase1_dno_losses = phase1_dno_losses
        self.prefix_values = (0, 20, obj_pose, obj_transl)
        self.table_corner_locs = table_new_corners
