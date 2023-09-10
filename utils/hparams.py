# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import yaml


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def get_hparams() -> HParams:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="./datasets/base/config.yaml", help="YAML file for configuration")
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name")
    args = parser.parse_args()

    # assert that path cnsists directory "datasets" and file "config.yaml
    assert os.path.exists("./datasets"), "`datasets` directory not found, navigate to the root of the project."
    assert os.path.exists(f"./datasets/{args.model}"), f"`{args.model}` not found in `./datasets/`"
    assert os.path.exists(f"./datasets/{args.model}/config.yaml"), f"`config.yaml` not found in `./datasets/{args.model}/`"

    model_dir = f"./datasets/{args.model}/logs"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config_path = args.config
    hparams = get_hparams_from_file(config_path)
    hparams.model_dir = model_dir
    return hparams


def get_hparams_from_file(config_path: str) -> HParams:
    with open(config_path, "r") as f:
        data = f.read()
    config = yaml.safe_load(data)

    hparams = HParams(**config)
    return hparams


if __name__ == "__main__":
    hparams = get_hparams()
    print(hparams)
