import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# =========================
# CONFIG
# =========================
IMG_SIZE = (768, 384)


# =========================
# FIND SEQUENCES
# =========================
def find_all_sequences(root_directories):
    sequences = []

    for root_dir in root_directories:
        for dirpath, _, filenames in os.walk(root_dir):
            if "CAM.csv" in filenames:
                sequences.append(dirpath)

    return sequences


# =========================
# PROCESS SINGLE ROW
# =========================
def process_row(args):
    root, row = args

    img_name = row["image_name"]
    pcd_name = row["pcd_name"]

    stitching_dir = depth_dir = seg_dir = latent_dir = None

    for d in os.listdir(root):
        full = os.path.join(root, d)
        if "stitching" in d:
            stitching_dir = os.path.join(full, "stitching")
        elif "depth" in d:
            depth_dir = os.path.join(full, "depth")
        elif "segmentation" in d:
            seg_dir = os.path.join(full, "segmentation")
        elif "latents" in d:
            latent_dir = full

    latent_path = os.path.join(latent_dir, pcd_name.replace(".pcd", ".npy"))
    stitch_path = os.path.join(stitching_dir, img_name)
    depth_path = os.path.join(depth_dir, img_name)
    seg_path = os.path.join(seg_dir, img_name)

    if not (os.path.exists(stitch_path) and
            os.path.exists(depth_path) and
            os.path.exists(seg_path) and
            os.path.exists(latent_path)):
        return None

    try:
        img_stitch = np.array(Image.open(stitch_path).resize(IMG_SIZE)) / 255.0
        img_depth = np.array(Image.open(depth_path).resize(IMG_SIZE)) / 255.0
        img_seg = np.array(Image.open(seg_path).resize(IMG_SIZE)) / 255.0

        if img_depth.ndim == 2:
            img_depth = img_depth[..., None]
        if img_seg.ndim == 2:
            img_seg = img_seg[..., None]

        x = np.concatenate([img_stitch, img_depth, img_seg], axis=2)
        x = torch.from_numpy(x).permute(2, 0, 1).float()

        y = torch.from_numpy(np.load(latent_path)).float()

        return x, y

    except:
        return None


# =========================
# PROCESS ONE SEQUENCE
# =========================
def process_sequence(root):

    cam_file = os.path.join(root, "CAM.csv")
    if not os.path.exists(cam_file):
        return []

    df = pd.read_csv(cam_file)

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(process_row, [(root, row) for _, row in df.iterrows()]),
            total=len(df),
            desc=f"Processing {os.path.basename(root)}"
        ))

    return [r for r in results if r is not None]


# =========================
# BUILD DATASET
# =========================
def build_dataset_parallel(sequence_dirs, chunk_size=1000, output_dir="dataset_chunks"):

    os.makedirs(output_dir, exist_ok=True)

    buffer_x = []
    buffer_y = []
    chunk_id = 0

    for root in tqdm(sequence_dirs, desc="Sequences"):

        samples = process_sequence(root)

        for x, y in samples:

            buffer_x.append(x)
            buffer_y.append(y)

            if len(buffer_x) >= chunk_size:
                torch.save(
                    (buffer_x, buffer_y),
                    os.path.join(output_dir, f"chunk_{chunk_id:04d}.pt")
                )
                print(f"[SAVE] chunk {chunk_id} ({len(buffer_x)} samples)")
                chunk_id += 1
                buffer_x, buffer_y = [], []

    # dernier chunk
    if len(buffer_x) > 0:
        torch.save(
            (buffer_x, buffer_y),
            os.path.join(output_dir, f"chunk_{chunk_id:04d}.pt")
        )
        print(f"[SAVE FINAL] chunk {chunk_id}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    root_datasets = [
        r"E:\PAIR360\Traversal2\College_of_Engineering",
        r"E:\PAIR360\Traversal2\College_of_Life_Science"
    ]

    print("Searching sequences...")
    sequence_directories = find_all_sequences(root_datasets)

    print(f"{len(sequence_directories)} sequences found")

    print("Building dataset in parallel...")
    build_dataset_parallel(
        sequence_directories,
        chunk_size=1000,
        output_dir="dataset_chunks"
    )

    print("Done")
