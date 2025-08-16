from typing import Optional
import torch
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes, knn_points
from pytorch3d.utils import ico_sphere


def points_inside_mesh(
    points: torch.Tensor,
    meshes: Meshes,
    ray_direction: torch.Tensor = torch.tensor([1.0, 0.0, 0.0]),
):
    """
    Determine for a batch of point clouds whether each point lies inside the corresponding
    closed mesh (using a ray–casting method).

    Args:
        points: Tensor of shape (B, N, 3) with 3D coordinates.
        meshes: A batched PyTorch3D Meshes object containing B meshes.
        ray_direction: A tensor of shape (3,) specifying the ray–casting direction.
                       (Same ray direction is used for all meshes and points.)

    Returns:
        inside: A boolean tensor of shape (B, N) where inside[b, n]==True means that the nth
                point in batch element b lies inside meshes[b].
    """
    device = points.device
    ray_direction = ray_direction.to(device).to(points.dtype)

    # Get padded vertices and faces. (B, V_max, 3) and (B, F_max, 3)
    verts_padded = meshes.verts_padded()  # shape: (B, V_max, 3)
    faces_padded = meshes.faces_padded()  # shape: (B, F_max, 3)

    # Create a mask indicating which faces are valid (all indices >= 0).
    valid_face_mask = (faces_padded >= 0).all(dim=2)  # shape: (B, F_max)

    # For each mesh in the batch, gather the triangle vertices.
    # Build batch indices for advanced indexing.
    B, F_max, _ = faces_padded.shape
    batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(B, F_max, 3)
    # triangles: (B, F_max, 3, 3) where each triangle is defined by 3 vertices (each 3D)
    triangles = verts_padded[batch_indices, faces_padded]

    # We now want to test each of the B point clouds (each with N points) against the
    # F_max triangles of the corresponding mesh.
    B, N, _ = points.shape
    # Expand points and triangles so that we can compute intersections:
    # points_exp: (B, N, 1, 3)
    points_exp = points.unsqueeze(2)
    # triangles_exp: (B, 1, F_max, 3, 3)
    triangles_exp = triangles.unsqueeze(1)

    # Broadcast ray_direction to shape (B, N, F_max, 3)
    D = ray_direction.view(1, 1, 1, 3).expand(B, N, F_max, 3)

    # Get triangle vertices V0, V1, V2.
    V0 = triangles_exp[..., 0, :]  # (B, 1, F_max, 3)
    V1 = triangles_exp[..., 1, :]  # (B, 1, F_max, 3)
    V2 = triangles_exp[..., 2, :]  # (B, 1, F_max, 3)

    # Compute edges.
    E1 = V1 - V0  # (B, 1, F_max, 3)
    E2 = V2 - V0  # (B, 1, F_max, 3)

    # Begin Möller–Trumbore algorithm.
    # Compute h = cross(D, E2) and a = dot(E1, h)
    h = torch.cross(D, E2, dim=-1)  # (B, N, F_max, 3)
    a = (E1 * h).sum(dim=-1)  # (B, N, F_max)
    eps = 1e-6
    valid = a.abs() > eps  # (B, N, F_max)

    # Compute f = 1/a for valid triangles.
    f = torch.zeros_like(a)
    f[valid] = 1.0 / a[valid]

    # Compute s = (ray origin - V0)
    s = points_exp - V0  # (B, N, F_max, 3)
    u = f * (s * h).sum(dim=-1)  # (B, N, F_max)

    # Compute q and v.
    q = torch.cross(s, E1, dim=-1)  # (B, N, F_max, 3)
    v = f * (D * q).sum(dim=-1)  # (B, N, F_max)

    # Compute distance along the ray.
    t = f * (E2 * q).sum(dim=-1)  # (B, N, F_max)

    # Determine valid intersections.
    intersect = (
        valid & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & ((u + v) <= 1.0) & (t > eps)
    )  # shape: (B, N, F_max)

    # Only count intersections from valid (i.e. non-padded) triangles.
    intersect = intersect & valid_face_mask.unsqueeze(1)

    # Count intersections per point.
    num_intersections = intersect.sum(dim=-1)  # (B, N)

    # A point is inside the mesh if the number of intersections is odd.
    inside = num_intersections % 2 == 1
    return inside


def compute_penetration(
    mesh_A: Meshes,
    mesh_B: Meshes,
    num_samples: Optional[int] = None,
    ray_direction: torch.Tensor = None,
    reduction: str = "mean",
):
    """
    For each pair of meshes in the batch, compute the maximal or mean penetration depth.
    (That is, for each mesh pair, we estimate the distance from the deepest or average interior
    point on mesh_A to the boundary of mesh_B.)
    """
    device = mesh_A.device
    ray_direction = (
        torch.tensor([1.0, 0.0, 0.0]) if ray_direction is None else ray_direction
    )
    ray_direction = ray_direction.to(device)

    # Sample points from mesh_A and mesh_B.
    if num_samples is None:
        points_A = mesh_A.verts_padded()
        points_B = mesh_B.verts_padded()
    else:
        points_A = sample_points_from_meshes(mesh_A, num_samples=num_samples)
        points_B = sample_points_from_meshes(mesh_B, num_samples=num_samples)

    # Determine, for each mesh in the batch, which sample points from mesh_A lie inside mesh_B.
    inside_mask = points_inside_mesh(points_A, mesh_B, ray_direction=ray_direction)

    # Find the distance to the nearest sampled point on mesh_B.
    knn = knn_points(points_A, points_B, K=1)
    distances = torch.sqrt(knn.dists.squeeze(-1) + 1e-8)

    # Mask distances for points inside mesh_B.
    masked_distances = torch.where(
        inside_mask, distances, torch.full_like(distances, -1.0)
    )

    # Compute the desired reduction.
    if reduction == "max":
        penetration, _ = masked_distances.max(dim=1)
    elif reduction == "mean":
        # Compute mean only over valid (non-negative) distances for each batch element.
        valid_mask = inside_mask
        valid_sum = torch.where(valid_mask, distances, torch.zeros_like(distances)).sum(
            dim=1
        )
        valid_count = valid_mask.sum(dim=1).float()
        penetration = torch.zeros_like(valid_sum)
        nonzero = valid_count > 0
        penetration[nonzero] = valid_sum[nonzero] / valid_count[nonzero]
    else:
        raise ValueError("Reduction method must be 'max' or 'mean'.")

    # Clamp to ensure non-negative penetration.
    penetration = torch.clamp(penetration, min=0.0)
    return penetration


if __name__ == "__main__":
    # Device.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create a single icosphere (a watertight mesh).
    base_mesh = ico_sphere(4, device=device)

    # For demonstration, we construct a batch of 2 mesh pairs.
    # For mesh_A (reference) we use two copies of base_mesh.
    # For mesh_B we use one unshifted copy and one offset copy.
    mesh_A_batch = Meshes(
        verts=[base_mesh.verts_packed(), base_mesh.verts_packed()],
        faces=[base_mesh.faces_packed(), base_mesh.faces_packed()],
    )

    # Create a second copy and then offset it to cause penetration.
    mesh_B_1 = base_mesh
    offset_vector = torch.tensor(
        [5.0, 0.0, 0.0], device=device
    )  # adjust offset to induce penetration
    mesh_B_2 = base_mesh.offset_verts(offset_vector)
    mesh_B_batch = Meshes(
        verts=[mesh_B_1.verts_packed(), mesh_B_2.verts_packed()],
        faces=[mesh_B_1.faces_packed(), mesh_B_2.faces_packed()],
    )

    # Compute the maximal penetration depth for each mesh pair in the batch.
    # (num_samples can be increased for a finer approximation.)
    max_penetration = compute_penetration(mesh_A_batch, mesh_B_batch, num_samples=2000)
    for i, pen in enumerate(max_penetration.tolist()):
        print(f"Mesh pair {i}: Maximal penetration depth = {pen:.4f}")
