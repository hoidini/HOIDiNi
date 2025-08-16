from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
import torch
import numpy as np
from copy import deepcopy
from scipy.spatial import distance_matrix as cdist

from hoidini.closd.diffusion_planner.utils import dist_util
from hoidini.cphoi.cphoi_utils import transform_smpldata
from hoidini.datasets.grab.grab_utils import (
    get_all_grab_seq_paths,
    get_df_grab_index,
    get_grab_split_ids,
    grab_seq_id_to_seq_path,
    grab_seq_path_to_seq_id,
)
from hoidini.datasets.smpldata import SmplData
from hoidini.object_contact_prediction.cpdm_dno_conds import (
    AboveTableLoss,
    KeepObjectStaticLoss,
    Specific6DoFLoss,
)
from hoidini.datasets.grab.grab_utils import load_mesh
from hoidini.datasets.grab.grab_utils import grab_seq_id_to_seq_path
from hoidini.datasets.smpldata_preparation import get_extended_smpldata
from hoidini.objects_fk import ObjectModel


@dataclass
class InferenceJob:
    grab_seq_path: Optional[str] = field(
        default=None, metadata={"help": "grab sequence path for prefix"}
    )
    text: Optional[str] = None
    object_name: Optional[str] = None
    start_frame: Optional[int] = None
    n_frames: int = 200
    start_frame: Optional[int] = None


class BaseSampler(ABC):
    @abstractmethod
    def get_inference_jobs(self) -> List[InferenceJob]:
        pass


class SpecificSampler(BaseSampler):
    def __init__(self, grab_seq_ids: List[str], n_reps: int = 1, n_frames: int = 115):
        super().__init__()
        self.grab_seq_ids = grab_seq_ids
        self.n_reps = n_reps
        self.n_frames = n_frames

    def get_inference_jobs(self) -> List[InferenceJob]:
        inference_jobs = []
        for grab_seq_id in self.grab_seq_ids:
            for i in range(self.n_reps):
                inference_jobs.append(
                    InferenceJob(
                        grab_seq_path=grab_seq_id_to_seq_path(grab_seq_id),
                        n_frames=self.n_frames,
                        start_frame=0,
                    )
                )
        return inference_jobs


class TestSampler(BaseSampler):
    def __init__(
        self,
        n_samples: int,
        split_name: str = "test",
        seed: int = 42,
        n_frames: Optional[int] = None,
        n_frames_lim: Optional[int] = 215,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.split_name = split_name
        self.seed = seed
        self.n_frames = n_frames
        self.n_frames_lim = n_frames_lim

    def get_inference_jobs(self) -> List[InferenceJob]:
        grab_split_ids = set(get_grab_split_ids(self.split_name))
        grab_seq_paths = get_all_grab_seq_paths()
        grab_seq_paths = [
            p for p in grab_seq_paths if grab_seq_path_to_seq_id(p) in grab_split_ids
        ]
        if self.n_samples != -1:
            grab_seq_paths = np.random.RandomState(self.seed).choice(
                grab_seq_paths, size=self.n_samples, replace=False
            )

        df_grab_index = get_df_grab_index()
        inference_jobs = []
        for grab_seq_path in grab_seq_paths:
            if self.n_frames is not None:
                n_frames = self.n_frames
            else:
                seq_id = grab_seq_path_to_seq_id(grab_seq_path)
                n_frames = df_grab_index.loc[seq_id].n_frames
                n_frames = min(n_frames, self.n_frames_lim)
            inference_jobs.append(
                InferenceJob(grab_seq_path=grab_seq_path, n_frames=n_frames)
            )
        return inference_jobs


class InTheWildObjectSampler(BaseSampler):
    def __init__(self, object_paths: int, split_name: str = "train", seed: int = 42):
        super().__init__()
        self.object_paths = object_paths
        self.split_name = split_name
        self.seed = seed

    def get_inference_jobs(self) -> List[InferenceJob]:
        grab_split_ids = set(get_grab_split_ids(self.split_name))
        grab_seq_paths = get_all_grab_seq_paths()
        grab_seq_paths = [
            p for p in grab_seq_paths if grab_seq_path_to_seq_id(p) in grab_split_ids
        ]
        grab_seq_paths = np.random.RandomState(self.seed).choice(
            grab_seq_paths, size=len(self.object_paths), replace=False
        )
        return [
            InferenceJob(grab_seq_path=grab_seq_path, object_name=object_path)
            for grab_seq_path, object_path in zip(grab_seq_paths, self.object_paths)
        ]


def get_corners(verts):
    """
    Assume the table is aligned
    """
    #  c4 -------- c3
    #  |           |
    #  |           |
    #  c1 -------- c2
    d1 = np.array([[-10, -10, 10]])
    d2 = np.array([[10, -10, 10]])
    d3 = np.array([[10, 10, 10]])
    d4 = np.array([[-10, 10, 10]])

    c1 = verts[cdist(verts, d1).argmin()]
    c2 = verts[cdist(verts, d2).argmin()]
    c3 = verts[cdist(verts, d3).argmin()]
    c4 = verts[cdist(verts, d4).argmin()]

    corners = np.stack([c1, c2, c3, c4])
    corners = torch.from_numpy(corners).float()
    return corners


def surface_loc_to_corners(xyz, dim=0.5):
    xyz = torch.tensor(xyz)
    dim = dim / 2
    c1 = xyz + torch.tensor([-dim, -dim, 0])
    c2 = xyz + torch.tensor([dim, -dim, 0])
    c3 = xyz + torch.tensor([dim, dim, 0])
    c4 = xyz + torch.tensor([-dim, dim, 0])
    return torch.stack([c1, c2, c3, c4])


def apply_transform(xyz, T):
    if xyz.shape[-1] != 3:
        raise ValueError("The last dimension of xyz must be 3.")
    if T.shape != (4, 4):
        raise ValueError("The transformation matrix T must be of shape (4, 4).")

    # Ensure xyz is a 2D tensor with shape [n, 3]
    original_shape = xyz.shape
    if len(original_shape) == 1:
        xyz = xyz.unsqueeze(0)  # Add batch dimension if it's just a single point

    xyz_h = torch.cat([xyz, torch.ones(xyz.shape[0], 1)], dim=1)
    transformed = T @ xyz_h.T
    result = transformed.T[:, :3]

    # Return to original shape if needed
    if len(original_shape) == 1:
        result = result.squeeze(0)

    return result


def inverse_transform(T):
    if T.shape != (4, 4):
        raise ValueError("The transformation matrix T must be of shape (4, 4).")
    return torch.linalg.inv(T)


SCRIPT_PAN = {
    "name": "cocking with the pan on the stove",
    "text": "The person is interacting with a fryingpan.",
    "object_name": "fryingpan",
    "surface_corners": surface_loc_to_corners(
        (-1.896400809288025, 0.49020278453826904, 1.0133963823318481), 0.4
    ),
    "start_frame": 0,
    "source_location": (-1.999293327331543 + 0.125, 0.49020278453826904, 0),
    "target_location": (-1.999293327331543 + 0.125, 0.49020278453826904, 0),
    "grab_seq_id": "s4/fryingpan_cook_2",
}

SCRIPT_BULB = {
    "name": "screwing a lightbulb",
    "text": "The person is screwing a lightbulb.",
    "grab_seq_id": "s3/lightbulb_screw_1",
    "object_name": "lightbulb",
    "source_location": (-0.1842409074306488, 1.1502920389175415, 0.85),
    "surface_corners": surface_loc_to_corners(
        (-0.1842409074306488, 1.1502920389175415, 0.85), 0.33
    ),
    # "target_location": (1.6560237407684326, 1.5103412866592407, 2.45),
}

SCRIPT_DRINK = {
    "name": "drinking",
    "text": "The person is drinking from a wineglass.",
    "grab_seq_id": "s1/wineglass_drink_2",
    "object_name": "wineglass",
    "location": (
        0.2952739894390106,
        -0.6584449410438538,
        0.9859883785247803 + 0.01,
    ),  # <--------------------------------------- add 0.01 to the z-coordinate to prevent visible penetration
    "surface_corners": surface_loc_to_corners(
        (0.2952739894390106, -0.6584449410438538, 0.9859883785247803 + 0.01), 0.33
    ),
}
# sample train grab seq for initial human pose and also object pose
# set the dno loss to keep the object static


class KitchenSampler(BaseSampler):
    pass


class KitchenSamplerPan(KitchenSampler):
    def __init__(self):

        grab_seq_id = SCRIPT_PAN["grab_seq_id"]
        grab_seq_path = grab_seq_id_to_seq_path(grab_seq_id)
        ext_smpldata = get_extended_smpldata(grab_seq_path)
        sd = ext_smpldata["smpldata"]

        # Grab Working Frame (G)
        # hmn_xyz_G = sd.trans[0]
        obj_xyz0_G = sd.trans_obj[0]
        obj_pose0_G = sd.poses_obj[0]

        # Kitchen Working Frame (K)
        obj_xyz0_K = torch.tensor(SCRIPT_PAN["source_location"])
        obj_xyz1_K = torch.tensor(SCRIPT_PAN["target_location"])
        # Rotation from kitchen to grab  ( ^|.  to  ^_. )

        obj_verts, obj_faces = load_mesh(SCRIPT_PAN["object_name"], 2000)
        # obj_fk = ObjectModel(v_template=obj_verts, batch_size=1)
        surface_corners = SCRIPT_PAN["surface_corners"]

        obj_xyz0_K[2] = surface_corners[:, 2].mean() - obj_verts[:, 2].min()
        obj_xyz1_K[2] = surface_corners[:, 2].mean() - obj_verts[:, 2].min()

        def zero_z(xyz):
            xyz = xyz.clone()
            xyz[..., 2] = 0
            return xyz

        R = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).float()
        T = torch.eye(4)
        T[:3, :3] = R
        T[:3, 3] = -R @ zero_z(obj_xyz0_K) + zero_z(obj_xyz0_G)

        obj_trans0_K_G = apply_transform(obj_xyz0_K, T)
        obj_trans1_K_G = apply_transform(obj_xyz1_K, T)
        surface_corners_G = apply_transform(surface_corners, T)

        self.rng1 = (0, 30)
        self.rng2 = (95, 115)

        self.obj_trans0_K_G = obj_trans0_K_G
        self.obj_trans1_K_G = obj_trans1_K_G
        self.global_orient1 = obj_pose0_G
        self.global_orient2 = obj_pose0_G

        self.transl1 = obj_trans0_K_G
        self.transl2 = obj_trans1_K_G
        self.surface_corners_G = surface_corners_G

        self.smpldata = sd
        self.phase1_dno_losses = {
            "above_table": (
                1.2,
                AboveTableLoss(surface_corners_G.unsqueeze(0)).to(dist_util.dev()),
            ),
            "keep_object_static": (1.2, KeepObjectStaticLoss()),
            "specific_6dof_table1": (
                0.5,
                Specific6DoFLoss(
                    self.transl1.unsqueeze(0),
                    self.global_orient1.unsqueeze(0),
                    target_frames=[torch.arange(self.rng1[0], self.rng1[1])],
                ).to(dist_util.dev()),
            ),
            "specific_6dof_table2": (
                0.5,
                Specific6DoFLoss(
                    self.transl2.unsqueeze(0),
                    self.global_orient2.unsqueeze(0),
                    target_frames=[torch.arange(self.rng2[0], self.rng2[1])],
                ).to(dist_util.dev()),
            ),
        }
        # self.phase2_dno_losses = {
        self.T = T  # use it's inverse to transform the entire generated sequence to the kitchen frame
        self.grab_seq_path = grab_seq_id_to_seq_path(grab_seq_id)

    def transform_smpldata(self, smpldata: SmplData):
        # transform the smpldata to the kitchen frame
        T = inverse_transform(self.T)
        rot_mat = T[:3, :3]
        trans_xyz = T[:3, 3]
        smpldata = transform_smpldata(smpldata, rot_mat, trans_xyz)
        return smpldata

    def get_phase1_dno_losses(self):
        return self.phase1_dno_losses

    def get_inference_jobs(self) -> List[InferenceJob]:
        inference_job = InferenceJob(
            grab_seq_path=self.grab_seq_path,
            text=SCRIPT_PAN["text"],
            object_name=SCRIPT_PAN["object_name"],
            n_frames=115,
            start_frame=0,
        )
        return [inference_job]


def zero_z(xyz):
    xyz = deepcopy(xyz)
    xyz[..., 2] = 0
    return xyz


class KitchenSamplerBulb(KitchenSampler):
    def __init__(self):

        grab_seq_id = SCRIPT_BULB["grab_seq_id"]
        grab_seq_path = grab_seq_id_to_seq_path(grab_seq_id)
        ext_smpldata = get_extended_smpldata(grab_seq_path)
        sd = ext_smpldata["smpldata"]

        # Grab Working Frame (G)
        # hmn_xyz_G = sd.trans[0]
        obj_xyz0_G = sd.trans_obj[0]
        obj_pose0_G = sd.poses_obj[0]
        # obj_pose0_G = torch.tensor([0, 0, 0])  # <---------------------------------------
        # Kitchen Working Frame (K)
        # obj_xyz0_K = torch.tensor(SCRIPT_BULB["source_location"])
        # obj_xyz1_K = torch.tensor(SCRIPT_BULB["target_location"])
        # Rotation from kitchen to grab  ( ^|.  to  ^_. )

        obj_verts, obj_faces = load_mesh(SCRIPT_BULB["object_name"], 2000)
        # obj_fk = ObjectModel(v_template=obj_verts, batch_size=1)
        surface_corners = SCRIPT_BULB["surface_corners"]  # (4, 3)
        # obj_xyz0_K[2] = surface_corners[:, 2].mean() - obj_verts[:, 2].min()
        # obj_xyz1_K[2] = surface_corners[:, 2].mean() - obj_verts[:, 2].min()

        # R = torch.tensor([
        #     [0, -1, 0],
        #     [1, 0, 0],
        #     [0, 0, 1]
        #     ]).float()
        # R = torch.eye(3)
        T = torch.eye(4)
        # T[:3, :3] = R
        # T[:3, 3] = - R @ zero_z(obj_xyz0_K) + zero_z(obj_xyz0_G)
        # T[:3, 3] = torch.zeros(3)

        # obj_trans0_K_G = apply_transform(obj_xyz0_K, T)
        # obj_trans1_K_G = apply_transform(obj_xyz1_K, T)
        self.surface_corners_G = apply_transform(surface_corners, T) + 99999

        self.rng1 = (0, 20)
        self.rng2 = (50, 70)
        self.n_frames = 115

        self.obj_trans0_K_G = obj_xyz0_G
        self.global_orient1 = obj_pose0_G
        self.global_orient2 = torch.tensor([0, torch.pi / 4, 0])
        self.transl1 = self.obj_trans0_K_G
        self.transl2 = torch.concat([obj_xyz0_G[:2], torch.tensor([1.7])])
        self.transl2 += torch.tensor([0.0, 0.3, 0]).to(self.transl2.device)

        self.smpldata = sd
        self.phase1_dno_losses = {
            # "above_table1": (alpha, AboveTableLoss(corners1.unsqueeze(0)).to(dist_util.dev())),
            # "above_table": (1.2, AboveTableLoss(self.surface_corners_G.unsqueeze(0)).to(dist_util.dev())),
            "keep_object_static": (0.6, KeepObjectStaticLoss()),
            # "specific_6dof_table1": (0.5, Specific6DoFLoss(self.transl1.unsqueeze(0), self.global_orient1.unsqueeze(0), target_frames=[torch.arange(self.rng1[0], self.rng1[1])]).to(dist_util.dev())),
            # "half_specific_6dof_bulb_socket": (0.5, get_half_specific_loss(self.transl2).to(dist_util.dev())),
            "specific_location": (
                0.6,
                Specific6DoFLoss(
                    self.transl2.unsqueeze(0),
                    self.global_orient1.unsqueeze(0),
                    target_frames=[torch.arange(self.rng2[0], self.rng2[1])],
                    location_only=True,
                ).to(dist_util.dev()),
            ),
            "bulb_up": (
                0.6,
                BulbUpLoss(target_frames=[torch.arange(self.rng2[0], self.rng2[1])]).to(
                    dist_util.dev()
                ),
            ),
        }
        self.T = T  # use it's inverse to transform the entire generated sequence to the kitchen frame
        self.grab_seq_path = grab_seq_id_to_seq_path(grab_seq_id)

    def transform_smpldata(self, smpldata: SmplData):
        # transform the smpldata to the kitchen frame
        T = inverse_transform(self.T)
        rot_mat = T[:3, :3]
        trans_xyz = T[:3, 3]
        smpldata = transform_smpldata(smpldata, rot_mat, trans_xyz)
        return smpldata

    def get_phase1_dno_losses(self):
        return self.phase1_dno_losses

    def get_inference_jobs(self) -> List[InferenceJob]:
        inference_jobs = []
        for i in range(20):
            inference_job = InferenceJob(
                grab_seq_path=grab_seq_id_to_seq_path(
                    np.random.choice(["s3/lightbulb_screw_1", "s9/lightbulb_screw_1"])
                ),
                text=SCRIPT_BULB["text"],
                object_name=SCRIPT_BULB["object_name"],
                n_frames=self.n_frames,
                start_frame=i,
            )
            inference_jobs.append(inference_job)
        return inference_jobs


class KitchenSamplerDrink(KitchenSampler):
    def __init__(self):

        grab_seq_id = SCRIPT_DRINK["grab_seq_id"]
        grab_seq_path = grab_seq_id_to_seq_path(grab_seq_id)
        ext_smpldata = get_extended_smpldata(grab_seq_path)
        sd = ext_smpldata["smpldata"]

        # Grab Working Frame (G)
        # hmn_xyz_G = sd.trans[0]
        obj_xyz0_G = sd.trans_obj[0]
        # obj_pose0_G = sd.poses_obj[0]
        obj_pose0_G = torch.tensor(
            [0, 0, 0]
        )  # <---------------------------------------
        # Kitchen Working Frame (K)
        obj_xyz0_K = torch.tensor(SCRIPT_DRINK["location"])
        obj_xyz1_K = torch.tensor(SCRIPT_DRINK["location"])
        # Rotation from kitchen to grab  ( ^|.  to  ^_. )

        obj_verts, obj_faces = load_mesh(SCRIPT_DRINK["object_name"], 2000)
        # obj_fk = ObjectModel(v_template=obj_verts, batch_size=1)
        surface_corners = SCRIPT_DRINK["surface_corners"]  # (4, 3)
        obj_xyz0_K[2] = surface_corners[:, 2].mean() - obj_verts[:, 2].min()
        obj_xyz1_K[2] = surface_corners[:, 2].mean() - obj_verts[:, 2].min()

        def zero_z(xyz):
            xyz = deepcopy(xyz)
            xyz[..., 2] = 0
            return xyz

        R = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).float()
        T = torch.eye(4)
        T[:3, :3] = R
        T[:3, 3] = -R @ zero_z(obj_xyz0_K) + zero_z(obj_xyz0_G)

        obj_trans0_K_G = apply_transform(obj_xyz0_K, T)
        obj_trans1_K_G = apply_transform(obj_xyz1_K, T)
        self.surface_corners_G = apply_transform(surface_corners, T)

        self.rng1 = (0, 20)
        self.rng2 = (95, 115)

        self.obj_trans0_K_G = obj_trans0_K_G
        self.obj_trans1_K_G = obj_trans1_K_G
        self.global_orient1 = obj_pose0_G
        self.global_orient2 = obj_pose0_G

        self.transl1 = obj_trans0_K_G
        self.transl2 = obj_trans1_K_G

        self.smpldata = sd
        self.phase1_dno_losses = {
            # "above_table1": (alpha, AboveTableLoss(corners1.unsqueeze(0)).to(dist_util.dev())),
            "above_table": (
                1.2,
                AboveTableLoss(self.surface_corners_G.unsqueeze(0)).to(dist_util.dev()),
            ),
            "keep_object_static": (1.2, KeepObjectStaticLoss()),
            "specific_6dof_table1": (
                0.5,
                Specific6DoFLoss(
                    self.transl1.unsqueeze(0),
                    self.global_orient1.unsqueeze(0),
                    target_frames=[torch.arange(self.rng1[0], self.rng1[1])],
                ).to(dist_util.dev()),
            ),
            "specific_6dof_table2": (
                0.5,
                Specific6DoFLoss(
                    self.transl2.unsqueeze(0),
                    self.global_orient2.unsqueeze(0),
                    target_frames=[torch.arange(self.rng2[0], self.rng2[1])],
                    rot_axis_to_include=[0, 1],
                ).to(dist_util.dev()),
            ),
        }
        self.T = T  # use it's inverse to transform the entire generated sequence to the kitchen frame
        self.grab_seq_path = grab_seq_id_to_seq_path(grab_seq_id)

    def transform_smpldata(self, smpldata: SmplData):
        # transform the smpldata to the kitchen frame
        T = inverse_transform(self.T)
        rot_mat = T[:3, :3]
        trans_xyz = T[:3, 3]
        smpldata = transform_smpldata(smpldata, rot_mat, trans_xyz)
        return smpldata

    def get_phase1_dno_losses(self):
        return self.phase1_dno_losses

    def get_inference_jobs(self) -> List[InferenceJob]:
        inference_job = InferenceJob(
            grab_seq_path=self.grab_seq_path,
            text=SCRIPT_DRINK["text"],
            object_name=SCRIPT_DRINK["object_name"],
            n_frames=115,
            start_frame=0,
        )
        return [inference_job]
