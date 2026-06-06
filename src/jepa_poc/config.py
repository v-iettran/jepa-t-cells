from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path = "configs/poc.yaml") -> DictConfig:
    return OmegaConf.load(path)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
