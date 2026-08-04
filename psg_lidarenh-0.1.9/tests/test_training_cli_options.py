from pathlib import Path


def test_training_cli_exposes_freeze_and_reinitialize_modes() -> None:
    source = (
        Path(__file__).parents[1]
        / "psg_lidarenh"
        / "training.py"
    ).read_text()
    assert '"--freeze-image-backbone"' in source
    assert '"--freeze-psgtr"' in source
    assert '"--reinitialize-psgtr-freeze-backbone"' in source
    assert ".from_psgtr_backbone_pretrained" in source
    assert "configure_trainability" in source
    assert "training-configuration.json" in source
