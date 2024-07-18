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

"""
if __name__ == "__main__":
    import time

    linear_layer = nn.Linear(1000, 1000, bias=True, dtype=torch.float16)
    rank = 10

    linear_layer.weight.data = torch.ones(1000, 1000, dtype=torch.float16)

    init_time = time.time()

    global_matrices_dict = nn.ModuleDict()
    average_matrices_dict = nn.ModuleDict()
    gbl = GlobalBaseLinear(
        linear_layer,
        global_matrices_dict,
        average_matrices_dict,
        target_name="query",
        rank=rank
    )

    print(f"Time taken: {(time.time() - init_time)}")

    print(gbl.weight.dtype)
    print(gbl.weight)

    print("Weights")
    print("Weights of the original layer")
    print(linear_layer.weight.shape)
    print()
    print("Weights of the global dependent layer")
    print(gbl_merged.weight)
    print()

    print(gbl)
    print(gbl_merged)
    print(gbl_merged.weight.data - linear_layer.weight.data)
    tolerance = 1e-7
    assert torch.allclose(gbl_merged.weight.data, linear_layer.weight.data, atol=tolerance)
   
    print("Output example")
    x = torch.ones(100, 100)
    print("Output of the original layer")
    print(linear_layer(x))
    print()
    print("Output of the global dependent layer")
    print(gbl(x))

"""