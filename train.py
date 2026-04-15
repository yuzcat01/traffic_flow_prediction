import argparse

from bootstrap import setup_project_paths

setup_project_paths()

from utils.config import load_yaml, merge_configs
from trainers.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", type=str, default="configs/data/pems04.yaml")
    parser.add_argument("--train_cfg", type=str, default="configs/train/default.yaml")
    parser.add_argument("--model_cfg", type=str, required=True)
    args = parser.parse_args()

    data_cfg = load_yaml(args.data_cfg)
    train_cfg = load_yaml(args.train_cfg)
    model_cfg = load_yaml(args.model_cfg)

    cfg = merge_configs(data_cfg, train_cfg, model_cfg)

    trainer = Trainer(cfg)
    trainer.train()
    trainer.test()


if __name__ == "__main__":
    main()
