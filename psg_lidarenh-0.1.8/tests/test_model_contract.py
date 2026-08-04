from pathlib import Path


def test_no_3d_box_code_is_packaged() -> None:
    source = (
        Path(__file__).parents[1]
        / "psg_lidarenh"
        / "modeling_psg_lidarenh.py"
    ).read_text()
    assert "boxes_3d" not in source
    assert "box_3d" not in source
    assert "relation_class_embed(fused_queries)" in source
    assert "subject_bbox_embed(fused_queries)" in source
    assert "object_bbox_embed(fused_queries)" in source
    assert "subject_logits\": base_output.subject_logits" in source
    assert "subject_masks\": base_output.subject_masks" in source
