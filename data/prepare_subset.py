import os
import kagglehub
import shutil


def prepare_imagenet100():
    script_dir = os.path.dirname(__file__)
    target_dir = os.path.abspath(os.path.join(script_dir, "..", "datasets", "subset_imagenet"))

    if os.path.isdir(target_dir) and os.listdir(target_dir):
        print(f"[INFO] Dataset already exists at '{target_dir}'. Skipping download.")
        return target_dir

    print("[INFO] Downloading ImageNet100 dataset from KaggleHub...")
    dataset_path = kagglehub.dataset_download("ambityga/imagenet100")

    print(f"[INFO] Downloaded to temporary location: {dataset_path}")
    print(f"[INFO] Moving dataset to: {target_dir}")
    shutil.move(dataset_path, target_dir)

    _cleanup_empty_dirs(os.path.dirname(dataset_path))

    print(f"[INFO] Done. Dataset available at: {target_dir}")
    return target_dir


def _cleanup_empty_dirs(start_path):
    path = os.path.abspath(start_path)
    while True:
        try:
            os.rmdir(path)
            path = os.path.dirname(path)
        except OSError:
            break


if __name__ == "__main__":
    prepare_imagenet100()