from pathlib import Path


def test_evaluation_calls_unwrapped_model_for_uneven_rank_partitions() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "psg_lidarenh"
        / "evaluation.py"
    ).read_text(encoding="utf-8")
    assert "outputs = unwrapped(" in source
    assert "outputs = model(" not in source


def test_training_prints_evaluation_progress() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "psg_lidarenh"
        / "training.py"
    ).read_text(encoding="utf-8")
    assert "evaluating train" in source
    assert "evaluating validation" in source
