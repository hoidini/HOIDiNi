import os
import pickle
from typing import List, Optional, Dict, Union
import torch
from torch import Tensor
import numpy as np
import torch.nn as nn
from pytorch3d.structures import Meshes

from hoidini.blender_utils.visualize_mesh import visualize_mesh
from hoidini.geometry3d.mesh_utils import (
    TopologyPreservingDecimation,
    numpy_to_open3d_mesh,
)
from hoidini.geometry3d.penetration_loss import compute_penetration
from hoidini.general_utils import SRC_DIR, torchify_numpy_dict
from hoidini.datasets.smpldata import SmplData
from hoidini.closd.diffusion_planner.utils import dist_util
from hoidini.datasets.smpldata import SmplModelsFK
from hoidini.blender_utils.visualize_mesh_figure_blender import get_smpl_template
from hoidini.skeletons.vertices_segments.vertex_ids import get_segments_vertex_ids_dict


DO_FLIP = {
    (True, True, False): True,
    (True, False, True): False,
    (False, True, True): True,
}

CACHE_DIR_PATH = "/home/dcor/roeyron/tmp"


def get_hand_meshes_building_data(
    smpl_model_name="smplx", n_simplify_faces: Optional[int] = 500
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    xxx_0 - w.r.t full body
    xxx_1 - w.r.t hand
    """
    body_segments_verts_indices_0 = get_segments_vertex_ids_dict(smpl_model_name)
    faces_0, verts_tpose_0 = get_smpl_template(smpl_model_name)
    faces_per_hand = {}
    vert_inds_wrt_body_per_hand = {}
    bracelet_vert_inds_wrt_body_per_hand = {}
    for hand in ["left", "right"]:
        hand_vert_inds_0 = np.array(
            body_segments_verts_indices_0[f"{hand}Hand"]
            + body_segments_verts_indices_0[f"{hand}HandIndex1"]
        )
        map_0_to_1 = np.vectorize(
            {ind_0: ind_1 for ind_1, ind_0 in enumerate(hand_vert_inds_0)}.get
        )
        # Cut the full body mesh to hand and reindex
        hand_faces_1 = map_0_to_1(
            faces_0[np.isin(faces_0, hand_vert_inds_0).sum(axis=1) == 3]
        )

        # Make the hand mesh watertight by adding new point on the center of the bracelet
        # Notice that the new point will be absent from future meshes and will have to be calculated
        bracelet_vert_inds_0 = np.array(
            sorted(
                set.intersection(
                    set(body_segments_verts_indices_0[f"{hand}ForeArm"]),
                    set(body_segments_verts_indices_0[f"{hand}Hand"]),
                )
            )
        )
        bracelet_vert_inds_1 = map_0_to_1(bracelet_vert_inds_0)
        bracelet_hand_faces_1 = hand_faces_1[
            np.isin(hand_faces_1, bracelet_vert_inds_1).sum(axis=1) >= 2
        ]
        is_face_vert_in_bracelet_mask = np.isin(
            bracelet_hand_faces_1, bracelet_vert_inds_1
        )
        new_vert_ind_1 = len(hand_vert_inds_0)
        new_hand_faces_1 = []
        for i, face_make in enumerate(is_face_vert_in_bracelet_mask):
            edge = bracelet_hand_faces_1[i][face_make]
            if DO_FLIP[tuple(face_make)]:  # make sure normals are pointing out
                edge = edge[::-1]
            new_hand_faces_1.append(np.concatenate([edge, [new_vert_ind_1]]))
        new_hand_faces_1 = np.stack(new_hand_faces_1)
        hand_faces_wt_1 = np.concatenate([hand_faces_1, new_hand_faces_1])

        if n_simplify_faces is not None:
            # create the T-pose watertight mesh for the topology simplification
            hand_verts_tpose_1 = verts_tpose_0[hand_vert_inds_0]
            new_vert_tpose_1 = (
                verts_tpose_0[bracelet_vert_inds_0].mean(axis=0).reshape(1, 3)
            )
            hand_verts_wt_1 = np.concatenate(
                [hand_verts_tpose_1, new_vert_tpose_1], axis=0
            )

            # visualize_mesh(hand_verts_wt_1, hand_faces_wt_1, save_blend_fname=f'./mesh_{hand}_watertight_orig.blend')
            mesh_o3d = numpy_to_open3d_mesh(hand_verts_wt_1, hand_faces_wt_1)
            decimator = TopologyPreservingDecimation()
            decimator.fit(mesh_o3d, target_triangles=n_simplify_faces)

            simp_hand_faces_wt_1 = decimator.final_faces
            hand_vert_inds_1 = decimator.alive_v_indices
            # visualize_mesh(hand_verts_wt_1[hand_vert_inds_1], simp_hand_faces_wt_1, save_blend_fname=f'./mesh_{hand}_watertight_simplified_{n_simplify_faces}.blend')
            hand_vert_inds_1 = hand_vert_inds_1[hand_vert_inds_1 != new_vert_ind_1]

            final_hand_faces = simp_hand_faces_wt_1
            final_body_to_hand_ind_map = hand_vert_inds_0[hand_vert_inds_1]
            final_body_to_bracelet_ind_map = np.array(
                [
                    i0
                    for i1, i0 in enumerate(bracelet_vert_inds_0)
                    if i1 in hand_vert_inds_1
                ]
            )
        else:
            final_hand_faces = hand_faces_wt_1
            final_body_to_hand_ind_map = hand_vert_inds_0
            final_body_to_bracelet_ind_map = bracelet_vert_inds_0

        faces_per_hand[hand] = final_hand_faces
        vert_inds_wrt_body_per_hand[hand] = final_body_to_hand_ind_map
        bracelet_vert_inds_wrt_body_per_hand[hand] = final_body_to_bracelet_ind_map
        print(f"{hand} hand #faces = {len(final_hand_faces)}")
    hand_meshes_building_data = {
        "faces_per_hand": faces_per_hand,
        "vert_inds_wrt_body_per_hand": vert_inds_wrt_body_per_hand,
        "bracelet_vert_inds_wrt_body_per_hand": bracelet_vert_inds_wrt_body_per_hand,
    }
    return hand_meshes_building_data


class HandIntersectionLoss(nn.Module):
    def __init__(
        self,
        smpl_model_name="smplx",
        n_simplify_faces_hands: Optional[int] = 800,
        device="cuda",
        use_cache=True,
    ):
        super().__init__()
        self.device = device
        cache_path = self.get_cache_path(smpl_model_name, n_simplify_faces_hands)
        if use_cache and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
        else:
            data = get_hand_meshes_building_data(
                smpl_model_name, n_simplify_faces_hands
            )
            if use_cache:
                with open(cache_path, "wb") as f:
                    pickle.dump(data, f)

        print(f"#### Using hand mesh with {data['faces_per_hand']['left'].shape[0]}")
        self.faces_per_hand = torchify_numpy_dict(data["faces_per_hand"], device)
        self.vert_inds_wrt_body_per_hand = torchify_numpy_dict(
            data["vert_inds_wrt_body_per_hand"], device
        )
        self.bracelet_vert_inds_wrt_body_per_hand = torchify_numpy_dict(
            data["bracelet_vert_inds_wrt_body_per_hand"], device
        )
        self.n_simplify_faces_hands = n_simplify_faces_hands

    @staticmethod
    def get_cache_path(smpl_model_name, n_simplify_faces):
        return os.path.join(
            CACHE_DIR_PATH,
            f"hand_meshes_building_data_{smpl_model_name}_{n_simplify_faces}.pickle",
        )

    def get_hand_mesh(self, hand, verts_batch: Tensor) -> Meshes:
        hand_verts_inds = self.vert_inds_wrt_body_per_hand[hand]
        bracelet_verts_inds = self.bracelet_vert_inds_wrt_body_per_hand[hand]
        hand_faces = self.faces_per_hand[hand]
        bs = verts_batch.shape[0]
        hand_faces = hand_faces.expand((bs,) + hand_faces.shape)
        # make the hand water-tight
        new_vert = verts_batch[:, bracelet_verts_inds].mean(dim=1)
        verts_hand_batch = verts_batch[:, hand_verts_inds]
        verts_hand_batch = torch.concatenate(
            [verts_hand_batch, new_vert.reshape(bs, -1, 3)], dim=1
        )
        mesh = Meshes(verts_hand_batch, hand_faces)
        return mesh

    def forward(
        self,
        body_verts: Tensor,
        object_verts: Optional[List[Tensor]] = None,
        object_faces: Optional[List[Tensor]] = None,
    ) -> Tensor:
        """
        body_verts.shape = (bs, seq, n_verts_body, 3)
        object_verts.shape = list of (seq, n_verts_obj, 3) with len = bs
        object_faces.shape = list of (n_faces, 3) with len = bs

        return losses.shape = (bs, seq_len)
        """

        bs, seq, n_verts_body, _ = body_verts.shape
        assert (object_verts is None) == (
            object_faces is None
        ), "Both object_verts and object_faces must be provided or none"

        if object_verts is not None:
            assert (
                len(object_verts) == bs
            ), f"Expected {bs} objects, got {len(object_verts)}"
            assert (
                isinstance(object_faces, list) and len(object_faces) == bs
            ), f"Expected {bs} face lists, got {len(object_faces)}"

            # Repeat each object seq times and combine
            losses = torch.zeros(bs, seq, device=self.device)
            for b in range(bs):
                losses[b] = self._forward(
                    body_verts[b],
                    object_verts[b],
                    object_faces[b].expand((seq,) + object_faces[b].shape),
                )

            # Calculate losses using lists for efficient Meshes creation
        else:
            body_verts = body_verts.view(bs * seq, n_verts_body, 3)
            losses = self._forward(body_verts, None, None)
            losses = losses.view(bs, seq)

        losses = losses.mean(dim=1)
        return losses

    def _forward(
        self,
        body_verts: Tensor,
        object_verts: Optional[Tensor] = None,
        object_faces: Optional[Tensor] = None,
    ) -> Tensor:
        """
        body_verts.shape = (bs, n_verts, 3)
        object_verts.shape = (bs, n_verts_obj, 3)
        object_faces.shape = (bs, n_faces, 3)

        losses.shape = (bs, )
        """
        mesh_left = self.get_hand_mesh("left", body_verts)
        mesh_right = self.get_hand_mesh("right", body_verts)
        losses = compute_penetration(mesh_left, mesh_right, num_samples=None)

        if object_verts is not None:
            mesh_object = Meshes(object_verts, object_faces)
            losses += compute_penetration(mesh_left, mesh_object, num_samples=None)
            losses += compute_penetration(mesh_right, mesh_object, num_samples=None)

        return losses


def main():
    dist_util.setup_dist(-1)

    ##########
    # Load test data
    ##########
    with open(os.path.join(SRC_DIR, "resources", "smpldata_lst.pkl"), "rb") as f:
        smpldata_lst: List[SmplData] = pickle.load(f)
    smpldata_lst = [e.to(dist_util.dev()) for e in smpldata_lst]
    min_len = min([len(smpldata) for smpldata in smpldata_lst])
    smpldata_lst = [smpldata.cut(0, min_len) for smpldata in smpldata_lst]
    smpl_fk = SmplModelsFK.create("smplx", len(smpldata_lst[0]), device=dist_util.dev())
    smpl_out_lst = smpl_fk.smpldata_to_smpl_output_batch(smpldata_lst)
    verts = smpl_out_lst[0].vertices[:120]
    ##########
    # Calculate hand intersection loss
    ##########
    hand_inters_loss_func = HandIntersectionLoss(
        device=dist_util.dev(), n_simplify_faces_hands=None
    )
    loss = hand_inters_loss_func(verts)
    print(f"negative example penetration loss = {loss[24].item():.4f}")
    print(f"positive example penetration loss = {loss[74].item():.4f}")

    import time

    tic = time.time()
    hand_inters_loss_func = HandIntersectionLoss(
        device=dist_util.dev(), n_simplify_faces_hands=400
    )
    print("delta", time.time() - tic)
    loss = hand_inters_loss_func(verts)
    print(f"negative example penetration loss = {loss[24].item():.4f}")
    print(f"positive example penetration loss = {loss[74].item():.4f}")


if __name__ == "__main__":
    main()
    # main_debug()


"""
cd /home/dcor/roeyron/trumans_utils/src
conda activate mahoi
export PYTHONPATH=/home/dcor/roeyron/trumans_utils/src
python geometry3d/hands_intersection_loss.py
python -m cProfile -o output.prof mesh_losses/hands_intersection_loss.py
snakeviz output.prof
"""
