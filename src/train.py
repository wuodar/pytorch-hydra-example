from pathlib import Path

import hydra
from omegaconf import DictConfig
import numpy as np
import random
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchmetrics


def train(
    model: nn.Module,
    metric: torchmetrics.Metric,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    epochs: int,
    cuda: bool,
    ckpt_save_path: Path,
):
    device = torch.device("cuda" if cuda else "cpu")
    model = model.to(device)
    metric = metric.to(device)

    for epoch in range(epochs):
        for batch_X, batch_y in train_dataloader:
            optimizer.zero_grad()
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            loss.backward()
            optimizer.step()
            metric.update(predictions.max(1)[1], batch_y)

        print(
            f"Epoch {epoch}, results on train dataset: "
            f"{loss.detach().cpu().numpy():.6f}, {metric.compute().cpu().numpy():.6f}"
        )
        metric.reset()

        for batch_X, batch_y in test_dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            metric.update(predictions.max(1)[1], batch_y)

        print(
            f"Epoch {epoch}, results on test dataset: "
            f"{loss.detach().cpu().numpy():.6f}, {metric.compute().cpu().numpy():.6f}"
        )
        metric.reset()

    ckpt = {"model": model, "optimizer": optimizer.state_dict()}
    torch.save(ckpt, ckpt_save_path)


def _get_transform(augs: DictConfig) -> transforms.Compose:
    augmentations: list[nn.Module] = []
    if augs:
        for augmentation_cfg in augs.values():
            augmentations.append(hydra.utils.instantiate(augmentation_cfg))
    return transforms.Compose(augmentations)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    if cfg.get("seed"):
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    train_transform = _get_transform(cfg.transform.train)
    train_dataset: datasets.VisionDataset = hydra.utils.instantiate(
        cfg.dataset.dataset, transform=train_transform, train=True, download=True
    )
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size)

    test_transform = _get_transform(cfg.transform.test)
    test_dataset: datasets.VisionDataset = hydra.utils.instantiate(
        cfg.dataset.dataset, transform=test_transform, train=False, download=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size)

    num_classes: int = cfg.dataset.num_classes
    model: nn.Module = hydra.utils.instantiate(cfg.model.net, output_size=num_classes)
    metric: torchmetrics.Metric = hydra.utils.instantiate(cfg.model.metric, num_classes=num_classes)
    optimizer: optim.Optimizer = hydra.utils.instantiate(cfg.model.optimizer, params=model.parameters())
    criterion: nn.Module = hydra.utils.instantiate(cfg.model.criterion)


    ckpt_save_path = Path(cfg.save_path) / "checkpoint.pt"
    ckpt_save_path.parent.mkdir(parents=True, exist_ok=True)

    train(
        model=model,
        metric=metric,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        epochs=cfg.epochs,
        cuda=cfg.cuda,
        ckpt_save_path=ckpt_save_path,
    )


if __name__ == "__main__":
    main()
