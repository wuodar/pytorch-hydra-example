from pathlib import Path
import random

import albumentations as A
import hydra
from omegaconf import DictConfig
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets
from torch.utils.data import DataLoader
import torchmetrics

from src.utils.transforms import AlbumentationsTransforms


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
        model.train()
        for batch_X, batch_y in train_dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            _, predictions = torch.max(outputs, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            metric.update(predictions, batch_y)

        print(
            f"Epoch {epoch}, results on train dataset: "
            f"Loss: {loss.item():.6f}, {metric.__class__.__name__}: {metric.compute().item():.6f}"
        )
        metric.reset()
        model.eval()

        with torch.inference_mode():
            for batch_X, batch_y in test_dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                _, predictions = torch.max(outputs, 1)

                metric.update(predictions, batch_y)

            print(
                f"Epoch {epoch}, results on test dataset: "
                f"Loss: {loss.item():.6f}, {metric.__class__.__name__}: {metric.compute().item():.6f}"
            )
        metric.reset()

    ckpt = {"model": model, "optimizer": optimizer.state_dict()}
    torch.save(ckpt, ckpt_save_path)


def _get_transforms(cfg: DictConfig):
    train_transform = hydra.utils.instantiate(cfg.transform.train)
    test_transform = hydra.utils.instantiate(cfg.transform.test)
    if isinstance(train_transform, A.Compose):
        train_transform = AlbumentationsTransforms(train_transform)
    if isinstance(test_transform, A.Compose):
        test_transform = AlbumentationsTransforms(test_transform)
    return train_transform, test_transform


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    if cfg.get("seed"):
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    train_transform, test_transform = _get_transforms(cfg)
    train_dataset: datasets.VisionDataset = hydra.utils.instantiate(
        cfg.dataset.dataset, transform=train_transform, train=True, download=True
    )
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size)

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
