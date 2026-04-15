import argparse

from src.project_paths import resolve_project_path
from src.trainers.trainer import Trainer
from src.utils.config import load_yaml, merge_configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", type=str, default="configs/data/pems04.yaml")
    parser.add_argument("--train_cfg", type=str, default="configs/train/default.yaml")
    parser.add_argument("--model_cfg", type=str, required=True)
    args = parser.parse_args()

    data_cfg = load_yaml(resolve_project_path(args.data_cfg))
    train_cfg = load_yaml(resolve_project_path(args.train_cfg))
    model_cfg = load_yaml(resolve_project_path(args.model_cfg))

    cfg = merge_configs(data_cfg, train_cfg, model_cfg)

    trainer = Trainer(cfg)
    trainer.train()
    trainer.test()


if __name__ == "__main__":
    main()
