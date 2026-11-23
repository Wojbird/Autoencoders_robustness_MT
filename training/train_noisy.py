from training._reconstruction import train_reconstruction_model


def train_noisy_model(
    model_class,
    config_path: str,
    *,
    dataset_type: str = "subset",
    log: bool = False,
    gpu_id: int | None = None,
    log_wandb: bool = False,
) -> str:
    return train_reconstruction_model(
        model_class,
        config_path,
        variant="noisy",
        dataset_type=dataset_type,
        log=log,
        gpu_id=gpu_id,
        log_wandb=log_wandb,
    )
