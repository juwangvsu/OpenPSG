from pathlib import Path


def test_evaluation_avoids_slow_mask_accumulator_update() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "psg_lidarenh"
        / "evaluation.py"
    ).read_text(encoding="utf-8")
    assert "accumulator.update(" not in source
    assert "_fast_accumulator_update(" in source
    assert "_panoptic_counts_from_maps(" in source
    assert "_predicate_recall_bbox_counts(" in source


def test_first_evaluation_sample_has_stage_progress() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "psg_lidarenh"
        / "evaluation.py"
    ).read_text(encoding="utf-8")
    assert "stage=forward" in source
    assert "stage=postprocess" in source
    assert "stage=metrics" in source
    assert "processed == 1" in source
