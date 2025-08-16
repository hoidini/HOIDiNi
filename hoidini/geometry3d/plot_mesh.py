import open3d as o3d
import numpy as np
import plotly.graph_objects as go


def plot_mesh(v=None, f=None, mesh: o3d.geometry.TriangleMesh = None):
    # Suppose we already have:
    #   mesh_in.vertices = o3d.utility.Vector3dVector(verts)
    #   mesh_in.triangles = o3d.utility.Vector3iVector(faces)

    if mesh is not None:
        v = np.asarray(mesh.vertices)
        f = np.asarray(mesh.triangles)

    # ----
    # 1) Create the main mesh trace (surface)
    # ----
    mesh_trace = go.Mesh3d(
        x=v[:, 0],
        y=v[:, 1],
        z=v[:, 2],
        i=f[:, 0],
        j=f[:, 1],
        k=f[:, 2],
        color="lightblue",
        opacity=0.8,
    )

    # ----
    # 2) Build edge information for a wireframe overlay
    # ----
    edges = set()
    for tri in f:
        # Each triangle has three edges: (i, j), (j, k), (k, i)
        for e in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            # Sort the edge so (1, 5) and (5, 1) aren't treated as duplicates
            edge = tuple(sorted(e))
            edges.add(edge)

    # Now create line coordinates for Plotly's Scatter3d
    edge_x = []
    edge_y = []
    edge_z = []

    for p1, p2 in edges:
        edge_x += [v[p1, 0], v[p2, 0], None]
        edge_y += [v[p1, 1], v[p2, 1], None]
        edge_z += [v[p1, 2], v[p2, 2], None]

    wireframe_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(color="black", width=1),
        showlegend=False,
    )

    # ----
    # 3) Add points trace with size 0
    # ----
    points_trace = go.Scatter3d(
        x=v[:, 0],
        y=v[:, 1],
        z=v[:, 2],
        mode="markers",
        marker=dict(size=0, color="black"),
        showlegend=False,
    )

    # ----
    # 4) Combine all traces in one figure
    # ----
    fig = go.Figure(data=[mesh_trace, wireframe_trace, points_trace])
    fig.update_layout(width=500, height=500, scene=dict(aspectmode="data"))
    fig.show()
