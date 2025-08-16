import pickle
from typing import List, Dict, Optional
import torch
from torch import Tensor
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from blender_utils.visualize_mesh import visualize_mesh
from geometry3d.mesh_utils import simplify_mesh
from geometry3d.penetration_loss import compute_penetration
from general_utils import torchify_numpy_dict
from datasets.smpldata import SmplData
from hoidini.closd.diffusion_planner.utils import dist_util
from datasets.smpldata import SmplModelsFK
from blender_utils.visualize_mesh_figure_blender import get_smpl_template
from skeletons.vertices_segments.vertex_ids import get_segments_vertex_ids_dict
from pytorch3d.structures import Meshes


DO_FLIP = {
    # since we're using existing faces, we use their order to deterimne the
    # order of the newly added faces
    (True, True, False): True,
    (True, False, True): False,
    (False, True, True): True,
}


def get_new_faces(old_faces, vert_inds_to_use):
    """
    Reduce and change vertices indices
    """
    relevant_old_faces = old_faces[np.isin(old_faces, vert_inds_to_use).sum(axis=1) == 3]
    old_to_new_ind_map = {old_ind: new_ind for new_ind, old_ind in enumerate(vert_inds_to_use)}
    new_faces = []
    for face in relevant_old_faces:
        new_faces.append([old_to_new_ind_map[ind] for ind in face])
    return np.array(new_faces)


def get_hand_verts_inds_dict(seg_vert_ids):
    hand_verts_inds_dict = {}
    for hand in ['left', 'right']:
        hand_verts_inds = seg_vert_ids[f'{hand}Hand'] + seg_vert_ids[f'{hand}HandIndex1']
        hand_verts_inds_dict[hand] = np.array(hand_verts_inds)
    return hand_verts_inds_dict


def get_hand_loop_vert_inds_dict(seg_vert_ids) -> Dict[str, np.ndarray[int]]:
    edge_vert_ids = {}
    for hand in ['left', 'right']:
        s1 = set(seg_vert_ids[f'{hand}ForeArm'])
        s2 = set(seg_vert_ids[f'{hand}Hand'])
        edge_vert_ids[hand] = np.array(sorted(set.intersection(s1, s2)))
    return edge_vert_ids


def get_hand_faces(faces_full_body, hand_verts_inds_dict, hand_loop_verts_inds_dict):
    hand_faces_dict = {}
    for hand in ['left', 'right']:
        hand_verts_old_inds = hand_verts_inds_dict[hand]
        hand_loop_verts_old_inds = hand_loop_verts_inds_dict[hand]

        old_to_new_ind_map = {old_ind: new_ind for new_ind, old_ind in enumerate(hand_verts_old_inds)}
        faces_hand = get_new_faces(faces_full_body, hand_verts_old_inds)
        hand_loop_verts_inds = [old_to_new_ind_map[i] for i in hand_loop_verts_old_inds]

        mask_loop_d1 = np.isin(faces_hand, hand_loop_verts_inds).sum(axis=1) >= 2  # (n_faces, )
        loop_faces = faces_hand[mask_loop_d1]
        mask_loop_d2 = np.isin(loop_faces, hand_loop_verts_inds)  # (n_faces, 3)
        new_vert_ind = len(hand_verts_old_inds)

        # hand new faces to make the hand water-tight
        new_lid_faces = []
        for i in range(len(mask_loop_d2)):
            sub_mask = mask_loop_d2[i]
            line = loop_faces[i][sub_mask]
            if DO_FLIP[tuple(sub_mask)]:
                line = line[::-1]
            face = np.concatenate([line, [new_vert_ind]])
            new_lid_faces.append(face)
        new_lid_faces = np.stack(new_lid_faces)
        faces_hand = np.concatenate([faces_hand, new_lid_faces])
        hand_faces_dict[hand] = faces_hand
        # hand_
    return hand_faces_dict


class HandIntersectionLoss(nn.Module):
    def __init__(self, smpl_model_name='smpl', device='cuda', n_simplify_faces: Optional[int] = 500):

        super(HandIntersectionLoss, self).__init__()
        self.device = device
        faces_full_body, verts_full_body = get_smpl_template(smpl_model_name)
        seg_vert_ids = get_segments_vertex_ids_dict(smpl_model_name)

        hand_verts_inds_dict = get_hand_verts_inds_dict(seg_vert_ids)           # Body world
        hand_loop_verts_inds_dict = get_hand_loop_vert_inds_dict(seg_vert_ids)  # Body world
        hand_faces_dict = get_hand_faces(faces_full_body, hand_verts_inds_dict, hand_loop_verts_inds_dict)  # Hand world

        if n_simplify_faces:
            for hand in ['left', 'right']:
                verts_hand = np.concatenate([
                    verts_full_body[hand_verts_inds_dict[hand]],
                    verts_full_body[hand_loop_verts_inds_dict[hand]].mean(axis=0).reshape(1, 3)
                    ], axis=0)
                faces_hand = hand_faces_dict[hand]
                # visualize_mesh(verts_hand, faces_hand)
                simplified_verts, simplified_faces, selected_vert_inds = simplify_mesh(verts_hand, faces_hand, tgt_faces=n_simplify_faces)
                # visualize_mesh(simplified_verts, simplified_faces)
                hand_verts_inds_dict[hand] = hand_verts_inds_dict[hand][selected_vert_inds]
                hand_faces_dict[hand] = simplified_faces

        #  used to map full body verts to hand verts  i_k: loc(k) is the new ind, val(i_k) is the old location
        self.hand_verts_inds_dict = torchify_numpy_dict(hand_verts_inds_dict, device)
        self.hand_loop_verts_inds_dict = torchify_numpy_dict(hand_loop_verts_inds_dict, device)
        self.hand_faces_dict = torchify_numpy_dict(hand_faces_dict, device)

    def get_mesh(self, hand, verts_batch: Tensor) -> Meshes:
        hand_verts_inds = self.hand_verts_inds_dict[hand]
        hand_loop_verts_inds = self.hand_loop_verts_inds_dict[hand]
        hand_faces = self.hand_faces_dict[hand]
        bs = verts_batch.shape[0]

        hand_faces = hand_faces.expand((bs,) + hand_faces.shape)
        # make the hand water-tight
        new_vert = verts_batch[:, hand_loop_verts_inds].mean(dim=1)
        verts_hand_batch = verts_batch[:, hand_verts_inds]
        verts_hand_batch = torch.concatenate([verts_hand_batch, new_vert.reshape(bs, -1 ,3)], dim=1)

        mesh = Meshes(verts_hand_batch, hand_faces)
        return mesh

    def forward(self, verts_batch: Tensor) -> Tensor:
        # verts_batch.shape = (batch_size, n_verts, 3)
        mesh_left = self.get_mesh('left', verts_batch)
        mesh_right = self.get_mesh('right', verts_batch)
        losses = compute_penetration(mesh_left, mesh_right, num_samples=None)
        return losses


def main():
    dist_util.setup_dist(-1)

    ##########
    # Load test data
    ##########
    with open('smpl_data_lst.pkl', 'rb') as f:
        smpl_data_lst: List[SmplData] = pickle.load(f)
    smpl_data_lst = [e.to(dist_util.dev()) for e in smpl_data_lst]
    smpl_fk = SmplModelsFK.create('smpl', len(smpl_data_lst[0]), device=dist_util.dev())
    smpl_out_lst = smpl_fk.smpldata_to_smpl_output_batch(smpl_data_lst)
    verts = smpl_out_lst[2].vertices[:75]
    ##########
    # Calcualte hand intersection loss
    ##########
    hand_inters_loss_func = HandIntersectionLoss(device=dist_util.dev(), n_simplify_faces=None)
    loss = hand_inters_loss_func(verts)
    print(f'negative example penetration loss = {loss[24].item():.4f}')
    print(f'positive example penetration loss = {loss[74].item():.4f}')

    hand_inters_loss_func = HandIntersectionLoss(device=dist_util.dev(), n_simplify_faces="all")
    loss = hand_inters_loss_func(verts)
    print(f'negative example penetration loss = {loss[24].item():.4f}')
    print(f'positive example penetration loss = {loss[74].item():.4f}')


def main_debug():
    ##########
    # Load test data
    ##########
    with open('smpl_data_lst.pkl', 'rb') as f:
        smpl_data_lst: List[SmplData] = pickle.load(f)
    smpl_data_lst = [e.to(dist_util.dev()) for e in smpl_data_lst]
    smpl_fk = SmplModelsFK.create('smpl', len(smpl_data_lst[0]), device=dist_util.dev())
    smpl_out_lst = smpl_fk.smpldata_to_smpl_output_batch(smpl_data_lst)

    # verts_batch = smpl_out_lst[2].vertices[:75]
    verts_hard = smpl_out_lst[2].vertices[[75]]
    verts_easy = smpl_out_lst[2].vertices[[24]]

    verts = verts_hard.expand(75, verts_hard.shape[1], verts_hard.shape[2])
    # verts = verts_easy.expand(75, verts_easy.shape[1], verts_easy.shape[2])

    verts = verts.clone()
    # verts_batch = torch.stack([smpl_out.vertices for smpl_out in smpl_out_lst])  # (batch_size, n_verts, 3)

    ##########
    # Calcualte hand intersection loss
    ##########
    hand_inters_loss_func = HandIntersectionLoss()
    loss = hand_inters_loss_func(verts)
    print(f'negative example penetration loss = {loss[24].item():.4f}')
    print(f'positive example penetration loss = {loss[74].item():.4f}')
    for i in tqdm(range(100)):
        loss = hand_inters_loss_func(verts)


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