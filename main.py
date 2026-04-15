from bootstrap import setup_project_paths

setup_project_paths()

from utils.config import load_yaml, merge_configs
from datasets.traffic_dataset import LoadData
from torch.utils.data import DataLoader


def main():
    data_cfg = load_yaml("configs/data/pems04.yaml")
    model_cfg = load_yaml("configs/model/chebnet_none.yaml")
    cfg = merge_configs(data_cfg, model_cfg)

    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]

    dataset = LoadData(
        data_path=[dataset_cfg["graph_path"], dataset_cfg["flow_path"]],
        num_nodes=dataset_cfg["num_nodes"],
        divide_days=dataset_cfg["divide_days"],
        time_interval=dataset_cfg["time_interval"],
        history_length=model_cfg["input"]["history_length"],
        train_mode="train",
        graph_type=model_cfg["graph"]["type"],
    )

    print("dataset len:", len(dataset))
    sample = dataset[0]
    print("graph shape:", sample["graph"].shape)
    print("flow_x shape:", sample["flow_x"].shape)
    print("flow_y shape:", sample["flow_y"].shape)

    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    print("batch graph shape:", batch["graph"].shape)
    print("batch flow_x shape:", batch["flow_x"].shape)
    print("batch flow_y shape:", batch["flow_y"].shape)


if __name__ == "__main__":
    main()
