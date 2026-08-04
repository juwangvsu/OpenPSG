import torch

from psg_lidarenh.lidar_encoder import GlobalLidarQueryFusion, LidarVoxelEncoder


def make_encoder() -> LidarVoxelEncoder:
    return LidarVoxelEncoder(
        4,
        16,
        hidden_dim=12,
        layers=2,
        point_cloud_range=(-10, -10, -3, 10, 10, 3),
        voxel_size=(1, 1, 1),
        max_tokens=8,
    )


def test_variable_clouds_become_padded_tokens() -> None:
    encoder = LidarVoxelEncoder(
        4,
        16,
        hidden_dim=12,
        layers=2,
        point_cloud_range=(-10, -10, -3, 10, 10, 3),
        voxel_size=(1, 1, 1),
        max_tokens=8,
    )
    first = torch.tensor([[0.1, 0.2, 0.3, 1.0], [1.2, 0.2, 0.3, 0.5]])
    second = torch.tensor([[2.1, 2.2, 0.3, 1.0]])
    tokens, valid = encoder([first, second])
    assert tokens.shape == (2, 2, 16)
    assert valid.tolist() == [[True, True], [True, False]]


def test_zero_gate_preserves_queries_exactly() -> None:
    fusion = GlobalLidarQueryFusion(
        16,
        4,
        ffn_dim=32,
        attention_dropout=0.0,
        dropout=0.0,
    )
    queries = torch.randn(2, 5, 16)
    tokens = torch.randn(2, 7, 16)
    valid = torch.ones(2, 7, dtype=torch.bool)
    output = fusion(queries, tokens, valid)
    assert torch.equal(output, queries)


def test_no_standalone_empty_token_parameter() -> None:
    encoder = make_encoder()
    parameter_names = dict(encoder.named_parameters())
    assert "empty_token" not in parameter_names


def test_empty_cloud_uses_token_mlp_parameters() -> None:
    encoder = make_encoder()
    tokens, _ = encoder([torch.empty((0, 4))])
    tokens.sum().backward()
    assert all(
        parameter.grad is not None
        for parameter in encoder.token_mlp.parameters()
    )
