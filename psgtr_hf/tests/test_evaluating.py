from __future__ import annotations

from pathlib import Path

import pytest

from psgtr_hf.evaluating import _report_name, parse_args, requested_splits


def test_evaluation_argument_defaults() -> None:
    args = parse_args(
        [
            "--checkpoint",
            "checkpoint-0004",
            "--data-root",
            "data/coco",
            "--annotation-file",
            "data/psg/psg_train_val.json",
            "--output-dir",
            "evaluation",
        ]
    )
    assert args.split == "both"
    assert args.samples == 200
    assert args.recall_k == [20, 50, 100]


def test_requested_splits_and_report_name() -> None:
    assert requested_splits("train") == ("train",)
    assert requested_splits("validation") == ("validation",)
    assert requested_splits("both") == ("train", "validation")
    assert _report_name(Path("checkpoint-0012")) == "evaluation-checkpoint-0012.json"


def test_invalid_sample_count_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--checkpoint",
                "checkpoint-0004",
                "--data-root",
                "data/coco",
                "--annotation-file",
                "data/psg/psg_train_val.json",
                "--output-dir",
                "evaluation",
                "--samples",
                "0",
            ]
        )
