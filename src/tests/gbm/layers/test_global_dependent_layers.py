import numpy as np

import torch
import torch.nn as nn

from gbm.layers.global_dependent_layer import LocalSVDLinear

class TestLocalSVDLinear:
    def test_LocalSVDLinear(self):
        linear_layer = nn.Linear(100, 100, bias=True)
        rank = 10
        global_matrices_dict = nn.ModuleDict()
        gbl = LocalSVDLinear(
            linear_layer,
            global_matrices_dict,
            rank=rank,
            target_name="query",
        )

        tolerance = 1e-7
        U, S, VT = np.linalg.svd(linear_layer.weight.data.numpy())
        min_dim = min(linear_layer.in_features, linear_layer.out_features)
        A = U[:, :min(min_dim, rank)] @ np.diag(S[:min(min_dim, rank)]) @ VT[:min(min_dim, rank), :]
        assert torch.allclose(
            torch.tensor(A).data,
            gbl.get_layer("local", "US").weight.data @ gbl.get_layer("local", "VT").weight.data,
            atol=tolerance
        )