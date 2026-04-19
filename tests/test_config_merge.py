import unittest

from src.utils.config import deep_update_dict, merge_configs


class ConfigMergeTests(unittest.TestCase):
    def test_merge_configs_preserves_nested_keys(self):
        merged = merge_configs(
            {
                "model": {
                    "temporal": {"type": "gru", "hidden_dim": 32, "num_layers": 1},
                    "output": {"predict_steps": 1},
                }
            },
            {
                "model": {
                    "temporal": {"type": "tcn"},
                    "output": {"predict_steps": 3},
                }
            },
        )

        self.assertEqual(merged["model"]["temporal"]["type"], "tcn")
        self.assertEqual(merged["model"]["temporal"]["hidden_dim"], 32)
        self.assertEqual(merged["model"]["temporal"]["num_layers"], 1)
        self.assertEqual(merged["model"]["output"]["predict_steps"], 3)

    def test_deep_update_dict_only_updates_given_keys(self):
        base = {
            "model": {
                "temporal": {"type": "tcn", "hidden_dim": 32, "kernel_size": 3, "num_layers": 2}
            }
        }
        deep_update_dict(base, {"model": {"temporal": {"hidden_dim": 64}}})

        self.assertEqual(base["model"]["temporal"]["type"], "tcn")
        self.assertEqual(base["model"]["temporal"]["hidden_dim"], 64)
        self.assertEqual(base["model"]["temporal"]["kernel_size"], 3)
        self.assertEqual(base["model"]["temporal"]["num_layers"], 2)


if __name__ == "__main__":
    unittest.main()
