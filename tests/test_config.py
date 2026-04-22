from __future__ import annotations

from pathlib import Path

from rve2d.config import load_config


def test_load_yaml_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "mode: synthetic",
                "synthetic:",
                "  domain_width: 1.0",
                "  domain_height: 1.0",
                "  fibre_radius: 0.05",
                "  target_volume_fraction: 0.2",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.mode == "synthetic"
    assert config.synthetic is not None
    assert config.synthetic.fibre_radius == 0.05


def test_load_sem_to_synthetic_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "mode: sem_to_synthetic",
                "sem_to_synthetic:",
                "  image_path: fibres.png",
                "  pixel_size: 1.5",
                "  smoothing_sigma: 1.0",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.mode == "sem_to_synthetic"
    assert config.sem_to_synthetic is not None
    assert config.sem_to_synthetic.pixel_size == 1.5
