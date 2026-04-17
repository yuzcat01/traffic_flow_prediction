import unittest

import torch

from src.models.builder import build_model
from src.models.spatial.graph_ops import build_binary_attention_mask, scaled_laplacian, symmetric_normalize_adjacency
from src.models.spatial.chebnet import ChebNetSpatial
from src.models.spatial.gat import GATSpatial
from src.models.spatial.gcn import GCNSpatial
from src.models.temporal.gru import GRUTemporal


class ModelBlockTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 2
        self.num_nodes = 5
        self.history_length = 4
        self.input_dim = 3
        self.hidden_dim = 8
        self.graph = torch.tensor(
            [
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        self.node_features = torch.randn(self.batch_size, self.num_nodes, self.input_dim)
        self.sequence_features = torch.randn(self.batch_size, self.num_nodes, self.history_length, self.hidden_dim)

    def test_gcn_forward_shape(self):
        model = GCNSpatial(in_c=self.input_dim, hidden_dim=self.hidden_dim, dropout=0.1)
        out = model(self.node_features, self.graph)
        self.assertEqual(out.shape, (self.batch_size, self.num_nodes, self.hidden_dim))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_chebnet_forward_shape(self):
        model = ChebNetSpatial(in_c=self.input_dim, hidden_dim=self.hidden_dim, K=3, dropout=0.1)
        out = model(self.node_features, self.graph)
        self.assertEqual(out.shape, (self.batch_size, self.num_nodes, self.hidden_dim))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_gat_forward_shape(self):
        model = GATSpatial(in_c=self.input_dim, hidden_dim=self.hidden_dim, heads=2, dropout=0.1)
        out = model(self.node_features, self.graph)
        self.assertEqual(out.shape, (self.batch_size, self.num_nodes, self.hidden_dim))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_gru_forward_shape(self):
        model = GRUTemporal(input_dim=self.hidden_dim, hidden_dim=12, num_layers=2, dropout=0.1)
        out = model(self.sequence_features)
        self.assertEqual(out.shape, (self.batch_size, self.num_nodes, 12))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_build_model_multistep_shape(self):
        cfg = {
            "model": {
                "name": "unit_test_model",
                "graph": {"type": "connect"},
                "input": {"history_length": self.history_length, "input_dim": 1},
                "spatial": {"type": "gcn", "hidden_dim": 8},
                "temporal": {"type": "gru", "hidden_dim": 16, "num_layers": 1},
                "regularization": {"dropout": 0.1},
                "output": {
                    "output_dim": 1,
                    "predict_steps": 3,
                    "head_type": "horizon_mlp",
                    "pred_hidden_dim": 32,
                    "horizon_emb_dim": 8,
                    "dropout": 0.1,
                    "use_last_value_residual": True,
                },
            }
        }
        model = build_model(cfg)
        batch = {
            "graph": self.graph,
            "flow_x": torch.randn(self.batch_size, self.num_nodes, self.history_length, 1),
        }
        out = model(batch)
        self.assertEqual(out.shape, (self.batch_size, self.num_nodes, 3, 1))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_graph_ops_keep_isolated_nodes_valid(self):
        directed_graph = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        symmetric_graph = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        mask = build_binary_attention_mask(directed_graph, symmetric=True, add_self_loop=True)
        norm_adj = symmetric_normalize_adjacency(symmetric_graph, add_self_loop=True)
        laplacian = scaled_laplacian(symmetric_graph, add_self_loop=True)

        self.assertEqual(mask.dtype, torch.bool)
        self.assertTrue(torch.equal(mask, mask.transpose(0, 1)))
        self.assertTrue(torch.equal(torch.diag(mask), torch.ones(3, dtype=torch.bool)))
        self.assertTrue(torch.isfinite(norm_adj).all().item())
        self.assertTrue(torch.isfinite(laplacian).all().item())
        self.assertTrue(torch.allclose(norm_adj, norm_adj.transpose(0, 1), atol=1e-6))
        self.assertTrue(torch.allclose(laplacian, laplacian.transpose(0, 1), atol=1e-6))

    def test_build_model_multistep_residual_falls_back_to_last_value(self):
        cfg = {
            "model": {
                "name": "unit_test_residual",
                "graph": {"type": "connect"},
                "input": {"history_length": self.history_length, "input_dim": 1},
                "spatial": {"type": "gcn", "hidden_dim": 4},
                "temporal": {"type": "gru", "hidden_dim": 6, "num_layers": 1},
                "regularization": {"dropout": 0.0},
                "output": {
                    "output_dim": 1,
                    "predict_steps": 3,
                    "head_type": "linear",
                    "use_last_value_residual": True,
                },
            }
        }
        model = build_model(cfg)
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()

        flow_x = torch.arange(
            1,
            1 + self.batch_size * self.num_nodes * self.history_length,
            dtype=torch.float32,
        ).view(self.batch_size, self.num_nodes, self.history_length, 1)
        batch = {
            "graph": self.graph,
            "flow_x": flow_x,
        }
        out = model(batch)
        expected = flow_x[:, :, -1:, :].repeat(1, 1, 3, 1)

        self.assertEqual(out.shape, expected.shape)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
