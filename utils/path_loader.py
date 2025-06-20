def load_imagenet_root(path_file="data_path.txt") -> str:
    try:
        with open(path_file, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise RuntimeError(f"File '{path_file}' with ImageNet path not found.")