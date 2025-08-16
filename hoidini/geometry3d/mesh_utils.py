import numpy as np
import open3d as o3d
from tqdm import tqdm
import pickle
from typing import List


def numpy_to_open3d_mesh(
    vertices: np.ndarray, faces: np.ndarray
) -> o3d.geometry.TriangleMesh:
    """
    Convert numpy arrays of vertices and faces to an Open3D TriangleMesh.
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    return mesh


def open3d_mesh_to_numpy(mesh: o3d.geometry.TriangleMesh):
    """
    Convert an Open3D TriangleMesh to numpy arrays (vertices, faces).
    """
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    return vertices, faces


class HalfEdgeMesh:
    """
    Maintains adjacency and supports edge-collapses:
      - self.vertices: (N,3) float array of 3D positions
      - self.faces: (M,3) int array of vertex indices
      - self.vertex_alive[v]: True/False
      - self.face_alive[f]: True/False
      - adjacency: self.neighbors[v] = set of adjacent vertices
      - self.incident_faces[v] = set of faces that contain vertex v
      - self.face_normals[f]
    """

    def __init__(self, vertices, faces):
        self.vertices = vertices.copy()
        self.faces = faces.copy()
        self.nv = self.vertices.shape[0]
        self.nf = self.faces.shape[0]

        # Build adjacency
        self.neighbors = [set() for _ in range(self.nv)]
        self.incident_faces = [set() for _ in range(self.nv)]
        for f_idx, f in enumerate(self.faces):
            v0, v1, v2 = f
            self.neighbors[v0].update([v1, v2])
            self.neighbors[v1].update([v0, v2])
            self.neighbors[v2].update([v0, v1])
            self.incident_faces[v0].add(f_idx)
            self.incident_faces[v1].add(f_idx)
            self.incident_faces[v2].add(f_idx)

        self.vertex_alive = np.ones(self.nv, dtype=bool)
        self.face_alive = np.ones(self.nf, dtype=bool)

        # Precompute face normals
        self.face_normals = np.zeros((self.nf, 3), dtype=float)
        self.compute_face_normals()

    def compute_face_normals(self):
        """Compute face normals for all alive faces."""
        for i in range(self.nf):
            if not self.face_alive[i]:
                continue
            v0, v1, v2 = self.faces[i]
            if (
                (not self.vertex_alive[v0])
                or (not self.vertex_alive[v1])
                or (not self.vertex_alive[v2])
            ):
                self.face_alive[i] = False
                continue
            p0 = self.vertices[v0]
            p1 = self.vertices[v1]
            p2 = self.vertices[v2]
            n = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(n)
            if norm < 1e-12:
                # degenerate face
                self.face_alive[i] = False
                self.face_normals[i] = np.array([0, 0, 0])
            else:
                self.face_normals[i] = n / norm

    def vertex_normal(self, v_idx):
        """
        Approximate normal at vertex v_idx by averaging normals of incident faces.
        """
        fset = [f for f in self.incident_faces[v_idx] if self.face_alive[f]]
        if not fset:
            return np.array([0, 0, 0], dtype=float)
        n_sum = np.zeros(3, dtype=float)
        for f_idx in fset:
            n_sum += self.face_normals[f_idx]
        norm = np.linalg.norm(n_sum)
        return n_sum / norm if norm > 1e-12 else n_sum

    def edge_cost(self, u, v):
        """
        Compute cost for collapsing edge (u,v).
        For demonstration: cost = edge_length * angle_between_vertex_normals(u,v).
        """
        p_u = self.vertices[u]
        p_v = self.vertices[v]
        edge_len = np.linalg.norm(p_u - p_v)

        n_u = self.vertex_normal(u)
        n_v = self.vertex_normal(v)
        dot = np.clip(n_u.dot(n_v), -1.0, 1.0)
        angle = np.arccos(dot)  # in [0, pi]

        return edge_len * angle

    def collapse_edge(self, u, v):
        """
        Collapse edge (u,v) by merging v into u.
        1) Move u to midpoint
        2) Mark v dead
        3) Update adjacency and remove degenerate faces
        """
        if (not self.vertex_alive[u]) or (not self.vertex_alive[v]):
            return

        # Move u to midpoint
        p_u = self.vertices[u]
        p_v = self.vertices[v]
        merged_pos = 0.5 * (p_u + p_v)
        self.vertices[u] = merged_pos

        # Update adjacency: neighbors of v become neighbors of u (except u itself)
        for nbr in self.neighbors[v]:
            if nbr != u:
                self.neighbors[nbr].discard(v)
                self.neighbors[nbr].add(u)
                self.neighbors[u].add(nbr)
        self.neighbors[u].discard(v)
        self.neighbors[v].clear()

        # Mark v as dead
        self.vertex_alive[v] = False

        # All faces referencing v become degenerate or must be updated
        faces_to_check = self.incident_faces[u].union(self.incident_faces[v])
        for f_idx in faces_to_check:
            if not self.face_alive[f_idx]:
                continue
            tri = self.faces[f_idx]
            if v in tri:
                # Replace v with u
                tri = [u if x == v else x for x in tri]
                if len(set(tri)) < 3:
                    self.face_alive[f_idx] = False
                else:
                    self.faces[f_idx] = tri

        self.incident_faces[v].clear()

        # Rebuild incident_faces[u]
        new_incident = set()
        for f_idx in faces_to_check:
            if self.face_alive[f_idx]:
                if u in self.faces[f_idx]:
                    new_incident.add(f_idx)
        for f_idx in self.incident_faces[u]:
            if self.face_alive[f_idx]:
                new_incident.add(f_idx)
        self.incident_faces[u] = new_incident

        # Recompute face normals for changed faces
        self.compute_face_normals()

    def decimate(self, target_faces):
        """
        Iteratively collapse edges of lowest cost until the number of alive faces <= target_faces.
        """
        n_faces = np.count_nonzero(self.face_alive)
        with tqdm(total=n_faces - target_faces, desc="Decimating") as pbar:
            while np.count_nonzero(self.face_alive) > target_faces:
                best_cost = 1e30
                best_edge = None

                # Naive approach: scan all adjacency to find the lowest-cost edge
                for u in range(self.nv):
                    if not self.vertex_alive[u]:
                        continue
                    for v in self.neighbors[u]:
                        if (not self.vertex_alive[v]) or (v < u):
                            continue
                        cost = self.edge_cost(u, v)
                        if cost < best_cost:
                            best_cost = cost
                            best_edge = (u, v)

                if best_edge is None:
                    # No more collapsible edges
                    break
                u, v = best_edge
                self.collapse_edge(u, v)

                delta = n_faces - np.count_nonzero(self.face_alive)
                n_faces = np.count_nonzero(self.face_alive)
                pbar.update(delta)

        return self.to_triangle_mesh()

    def to_triangle_mesh(self):
        """
        Convert alive vertices/faces to arrays. Also return 'alive_v_indices'.
        """
        alive_v_indices = np.nonzero(self.vertex_alive)[0]
        # old->new
        mapping = {}
        for i, old_idx in enumerate(alive_v_indices):
            mapping[old_idx] = i

        new_vertices = self.vertices[alive_v_indices]

        alive_f_indices = np.nonzero(self.face_alive)[0]
        new_faces_list = []
        for f_idx in alive_f_indices:
            tri = self.faces[f_idx]
            if not all(self.vertex_alive[v] for v in tri):
                continue
            new_tri = [mapping[v] for v in tri]
            if len(set(new_tri)) == 3:
                new_faces_list.append(new_tri)
        new_faces = np.array(new_faces_list, dtype=int)

        return new_vertices, new_faces, alive_v_indices


class TopologyPreservingDecimation:
    """
    Demonstrates a half-edge-based decimator with a fit/transform interface.
    """

    def __init__(self):
        self.alive_v_indices = None  # which reference vertices survived
        self.final_faces = (
            None  # final connectivity referencing [0..len(alive_v_indices)-1]
        )

    def fit(
        self, reference_mesh: o3d.geometry.TriangleMesh, target_triangles: int = 400
    ) -> o3d.geometry.TriangleMesh:
        """
        Decimate the reference Open3D mesh until face count <= target_triangles.

        Returns an Open3D mesh of the decimated reference.
        """
        # Convert to numpy
        ref_verts = np.asarray(reference_mesh.vertices)
        ref_faces = np.asarray(reference_mesh.triangles)
        n_faces = ref_faces.shape[0]
        if target_triangles >= n_faces:
            # No decimation needed
            self.alive_v_indices = np.arange(ref_verts.shape[0])
            self.final_faces = ref_faces.copy()
            return reference_mesh

        # Build half-edge mesh and decimate
        hemesh = HalfEdgeMesh(ref_verts, ref_faces)
        dec_verts, dec_faces, alive_idx = hemesh.decimate(target_triangles)

        # Store the pattern
        self.alive_v_indices = alive_idx
        self.final_faces = dec_faces

        # Build and return an Open3D mesh
        dec_mesh = o3d.geometry.TriangleMesh()
        dec_mesh.vertices = o3d.utility.Vector3dVector(dec_verts)
        dec_mesh.triangles = o3d.utility.Vector3iVector(dec_faces)
        dec_mesh.compute_vertex_normals()
        return dec_mesh

    def transform(
        self, new_mesh: o3d.geometry.TriangleMesh
    ) -> o3d.geometry.TriangleMesh:
        """
        Apply the decimation pattern to a new mesh that has the same vertex ordering
        as the reference. We select the same subset of vertices and reuse the final connectivity.
        """
        if self.alive_v_indices is None or self.final_faces is None:
            raise RuntimeError("Must call fit() before transform().")

        new_verts = np.asarray(new_mesh.vertices)
        # Subset the vertices
        dec_verts = new_verts[self.alive_v_indices]

        # Reuse the final faces
        dec_faces = self.final_faces.copy()

        # Build an Open3D mesh
        dec_mesh = o3d.geometry.TriangleMesh()
        dec_mesh.vertices = o3d.utility.Vector3dVector(dec_verts)
        dec_mesh.triangles = o3d.utility.Vector3iVector(dec_faces)
        dec_mesh.compute_vertex_normals()
        return dec_mesh


if __name__ == "__main__":
    from datasets.smpldata import SmplData
    from hoidini.closd.diffusion_planner.utils import dist_util
    from datasets.smpldata import SmplModelsFK
    from geometry3d.hands_intersection_loss import HandIntersectionLoss
    from geometry3d.plot_mesh import plot_mesh

    with open("smpl_data_lst.pkl", "rb") as f:
        smpl_data_lst: List[SmplData] = pickle.load(f)
    smpl_data_lst = [e.to(dist_util.dev()) for e in smpl_data_lst]
    smpl_fk = SmplModelsFK.create("smpl", len(smpl_data_lst[0]), device=dist_util.dev())
    smpl_out_lst = smpl_fk.smpldata_to_smpl_output_batch(smpl_data_lst)

    hand_inter_loss = HandIntersectionLoss(n_simplify_faces_hands=None)

    meshes0 = hand_inter_loss.get_hand_mesh("left", smpl_out_lst[2].vertices[[23]])
    verts0 = meshes0.verts_list()[0].detach().cpu().numpy()
    faces0 = meshes0.faces_list()[0].detach().cpu().numpy()

    meshes1 = hand_inter_loss.get_hand_mesh("left", smpl_out_lst[2].vertices[[74]])
    verts1 = meshes1.verts_list()[0].detach().cpu().numpy()
    faces1 = meshes1.faces_list()[0].detach().cpu().numpy()
    # -----------------------------------------------------------------------------
    # Example usage matching your code snippet
    # -----------------------------------------------------------------------------
    # Suppose you have numpy arrays verts0, faces0, verts1, faces1
    # Already defined somewhere in your code. For example:
    # verts0 = ...
    # faces0 = ...
    # verts1 = ...
    # faces1 = ...

    # 1) Convert numpy arrays to Open3D meshes:
    mesh_o3d_0 = numpy_to_open3d_mesh(verts0, faces0)
    mesh_o3d_1 = numpy_to_open3d_mesh(verts1, faces1)

    # 2) Create the decimator and learn from the reference mesh.
    decimator = TopologyPreservingDecimation()
    mesh_o3d_0_fit = decimator.fit(mesh_o3d_0, target_triangles=600)

    # 3) Convert the decimated reference mesh back to numpy (if needed)
    verts0_simpl_fit, faces0_simpl_fit = open3d_mesh_to_numpy(mesh_o3d_0_fit)

    # 4) Apply the learned decimation pattern to both the reference mesh and the new mesh.
    mesh_o3d_0_transformed = decimator.transform(mesh_o3d_0)
    mesh_o3d_1_transformed = decimator.transform(mesh_o3d_1)

    verts0_simpl_pred, faces0_simpl_pred = open3d_mesh_to_numpy(mesh_o3d_0_transformed)
    verts1_simpl_pred, faces1_simpl_pred = open3d_mesh_to_numpy(mesh_o3d_1_transformed)

    to_show = [
        ("origianl", verts0, faces0),
        ("simplified_fit", verts0_simpl_fit, faces0_simpl_fit),
        ("verts0_simpl_pred", verts0_simpl_pred, faces0_simpl_pred),
        ("verts1", verts1, faces1),
        ("verts1_simpl_pred", verts1_simpl_pred, faces1_simpl_pred),
    ]
    for txt, v, f in to_show:
        print(txt, v.shape, f.shape)
        plot_mesh(v, f)
