from pathlib import Path


def test_training_evaluation_is_rank_zero_only() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "psg_lidarenh"
        / "training.py"
    ).read_text(encoding="utf-8")
    assert "epoch % args.eval_every == 0 and rank == 0" in source
    assert "reduce_across_processes=False" in source
    assert 'f"samples={len(train_eval)} on rank=0"' in source


def test_evaluator_can_disable_collectives() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "psg_lidarenh"
        / "evaluation.py"
    ).read_text(encoding="utf-8")
    assert "reduce_across_processes: bool = True" in source
    assert "if reduce_across_processes and dist.is_available()" in source
    assert "progress_label" in source
