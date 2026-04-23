import open3d as o3d
import numpy as np
import os

directory = "E:\PAIR360\Traversal2\\1st_Dormitory\Sequence1\T2-1st_Dormitory-1-pcd\pcd"

i = 0

for filename in os.listdir(directory):

    i += 1

    if i % 50 != 0:
        continue

    if not filename.endswith(".pcd"):
        continue

    file_path = os.path.join(directory, filename)

    # Charger le PCD
    pcd = o3d.io.read_point_cloud(file_path)

    # Convertir en numpy (optionnel)
    points = np.asarray(pcd.points)

    print(f"{filename} → {points.shape[0]} points")

    # (optionnel) filtrer les points invalides
    points = points[~np.isnan(points).any(axis=1)]

    # remettre dans Open3D si modifié
    pcd.points = o3d.utility.Vector3dVector(points)

    # Visualisation
    o3d.visualization.draw_geometries([pcd])
