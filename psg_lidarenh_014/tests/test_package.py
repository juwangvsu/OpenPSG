from pathlib import Path


def test_required_modules_and_examples_are_present() -> None:
    root = Path(__file__).parents[1]
    required = (
        "psg_lidarenh/configuration_psg_lidarenh.py",
        "psg_lidarenh/dataset.py",
        "psg_lidarenh/evaluation.py",
        "psg_lidarenh/evaluating.py",
        "psg_lidarenh/kitti360.py",
        "psg_lidarenh/lidar_encoder.py",
        "psg_lidarenh/modeling_psg_lidarenh.py",
        "psg_lidarenh/training.py",
        "examples/convert_kitti360.py",
        "examples/evaluate.py",
        "examples/train.py",
    )
    for relative in required:
        assert (root / relative).is_file(), relative
