import open3d as o3d
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
# PCD_FOLDER = "./outputs"
PCD_FOLDER = r"E:\PAIR360\Traversal2\College_of_Physical_Education\Sequence0\T2-College_of_Physical_Education-0-pcd\pcd"
N_POINTS = 44000


def load_pcd_files(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.pcd")))
    if not files:
        raise ValueError(f"Aucun fichier .pcd trouvé dans {folder}")
    return files


# =========================
# SOUS-ÉCHANTILLONNAGE
# =========================
def subsample_pcd(pcd, n_points=4096):
    points = np.asarray(pcd.points)

    if len(points) == 0:
        return pcd

    # Si trop petit → duplication
    if len(points) < n_points:
        idx = np.random.choice(len(points), n_points, replace=True)
    else:
        idx = np.random.choice(len(points), n_points, replace=False)

    return pcd.select_by_index(idx)


# =========================
# COLORISATION HAUTEUR
# =========================
def colorize_by_height(pcd):
    points = np.asarray(pcd.points)

    if points.shape[0] == 0:
        return pcd

    z = points[:, 2]

    z_min, z_max = z.min(), z.max()
    z_norm = (z - z_min) / (z_max - z_min + 1e-8)

    colors = plt.cm.viridis(z_norm)[:, :3]

    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


# =========================
# VISUALISATION
# =========================
def visualize_pcd(file_path):
    print(f"Affichage : {file_path}")

    pcd = o3d.io.read_point_cloud(file_path)

    if len(pcd.points) == 0:
        print("⚠️ Nuage vide")
        return

    # 🔥 downsample à 4096 points
    pcd = subsample_pcd(pcd, N_POINTS)

    # couleur par hauteur
    pcd = colorize_by_height(pcd)

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=os.path.basename(file_path)
    )


# =========================
# MAIN
# =========================
def main():
    files = load_pcd_files(PCD_FOLDER)

    print(f"{len(files)} fichiers trouvés")

    for i, file in enumerate(files):
        print(f"[{i+1}/{len(files)}] {file}")
        visualize_pcd(file)
        input("ENTER pour suivant...")


if __name__ == "__main__":
    main()
