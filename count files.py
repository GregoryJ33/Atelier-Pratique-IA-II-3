import os
import torch

print(torch.__version__)

root = r"E:\PAIR360\Traversal2"

for subdir in os.listdir(root):
    subdir_path = os.path.join(root, subdir)

    if os.path.isdir(subdir_path):
        count = 0

        for dirpath, _, filenames in os.walk(subdir_path):
            count += sum(1 for f in filenames if f.lower().endswith(".pcd"))

        print(f"{subdir} : {count} fichiers .pcd")


root = r"E:\PAIR360"

for subdir in os.listdir(root):
    subdir_path = os.path.join(root, subdir)

    if os.path.isdir(subdir_path):
        count = 0

        for dirpath, _, filenames in os.walk(subdir_path):
            count += sum(1 for f in filenames if f.lower().endswith(".pcd"))

        print(f"{subdir} : {count} fichiers .pcd")