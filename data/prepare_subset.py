import os
import kagglehub
import shutil

def prepare_imagenet100():
    """
        Funkcja – Pobiera dane do .cache/kagglehub/ i przenosi je do datasets/subset_imagenet
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(script_dir, "..", "datasets", "subset_imagenet")

    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
        print(f"Dane już istnieją w '{target_dir}'. Pomijam pobieranie.")
        return target_dir

    print("Pobieram ImageNet100 z Kaggle...")
    dataset_path = kagglehub.dataset_download("ambityga/imagenet100")

    print(f"Dane pobrane do tymczasowej lokalizacji: {dataset_path}")

    print(f"Przenoszę dane do katalogu projektu: {target_dir}...")
    shutil.move(dataset_path, target_dir)

    _cleanup_empty_kagglehub_dirs(dataset_path)

    print("Gotowe! Dane dostępne w:", target_dir)
    return target_dir


def _cleanup_empty_kagglehub_dirs(old_path):
    """
    Pomocnicza funkcja – usuwa puste katalogi po przeniesieniu danych z .cache/kagglehub/
    """
    current = os.path.abspath(old_path)
    while True:
        parent = os.path.dirname(current)
        try:
            os.rmdir(current)
            current = parent
        except OSError:
            break  # katalog nie jest pusty lub nie da się usunąć


if __name__ == "__main__":
    prepare_imagenet100()