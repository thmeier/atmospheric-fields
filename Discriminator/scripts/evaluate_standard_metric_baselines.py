"""Evaluate standard metric baselines and persist plot-ready artifacts."""

import hydra
from omegaconf import DictConfig

try:
    from .plot_standard_metric_baselines import evaluate_standard_metrics
except ImportError:
    from plot_standard_metric_baselines import evaluate_standard_metrics


@hydra.main(version_base=None, config_path="../conf", config_name="baseline_config")
def main(cfg: DictConfig):
    evaluate_standard_metrics(cfg)


if __name__ == "__main__":
    main()
