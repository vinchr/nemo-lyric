import argparse
import json
import os.path
import subprocess

import pandas as pd

from ruamel.yaml import YAML

from omegaconf import OmegaConf
from omegaconf import DictConfig


# Script to run training jobs with nemo_quartznet_train.py on inputs from a csv file
# Example usage:
# python run_trainers.py <csvfile>
# where 'csvfile' looks like the following:
#
# training data is in train.json
# validation data is in validation.json
# hyper-params / model architecture yaml file is: models/dec14/quartznet_15x5_22k_adam_cosine.yaml
# Run for 10 epochs
# No checkpoint file to resume from
# extra args: --batch-size 16
# additional naming description: training_97
#
# Then the csv file would contain the following:
#
# train,validation,subdir,name,epochs,resume,args,desc
# train.json,validation.json,dec14,22k_adam_cosine,10,,--batch-size 16,training_97


def run_single_trainer(row, models_dir, out_dir, err_dir):
    # print("row dtypes", row.dtypes)
    name = ".".join(filter(None, [row["name"], row["desc"]]))
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(err_dir, exist_ok=True)
    found_idx = False
    for idx in range(100):
        name_idx = "{}.{:02d}".format(name, idx)
        nemo_file = os.path.join(models_dir, "quartznet_15x5_" + name_idx + ".nemo")
        out_file = os.path.join(out_dir, name_idx + ".log")
        err_file = os.path.join(err_dir, name_idx + ".err")
        if (os.path.exists(nemo_file) or
                os.path.exists(out_file) or
                os.path.exists(err_file)):
            continue
        found_idx = True
        break
    assert found_idx, name
    script = "nemo_quartznet_train.py"
    assert os.path.exists(script)
    model_cfg_filename = "quartznet_15x5_" + row["name"] + ".yaml"
    model_cfg_filepath = os.path.join("models", row["subdir"], model_cfg_filename)
    assert os.path.exists(model_cfg_filepath), "No such file: " + model_cfg_filepath
    cmd_list = ["python", script]
    cmd_list.extend(["--model", model_cfg_filepath])
    cmd_list.extend(["--train-ds", row["train"]])
    if row["validation"]:
        cmd_list.extend(("--val-ds", row["validation"]))
    cmd_list.extend(["--save", nemo_file])
    assert row["epochs"]
    cmd_list.extend(("--epochs", str(row["epochs"])))
    if row["resume"]:
        cmd_list.extend(("--resume-from-checkpoint", row["resume"]))
    cmd_list.extend(row["args"].split())
    print("Running:", " ".join(cmd_list))
    print("stdout: ", out_file)
    print("stderr: ", err_file)
    with open(out_file, "w") as out_handle:
        with open(err_file, "w") as err_handle:
            subprocess.run(cmd_list, stdout=out_handle, stderr=err_handle, check=True)
    print("Finished:", " ".join(cmd_list))


def run_trainers(spec_file):
    allowed_columns = ("train", "validation", "subdir", "name",
                       "epochs", "resume",
                       "args", "desc")
    # dtypes = {"train": str,
    #           "validation": str,
    #           "subdir": str,
    #           "name": str,
    #           "args": str,
    #           "desc": str}
    train_dir = "train"
    models_dir = os.path.join(train_dir, "models")
    out_dir = os.path.join(train_dir, "out")
    err_dir = os.path.join(train_dir, "err")
    specs_df = pd.read_csv(spec_file, na_filter=False, comment='#')
    # for idx, dtype in enumerate(specs_df.dtypes):
    #     if dtype == 'object':
    #         specs_df.iloc[:, idx] = specs_df.iloc[:, idx].astype('str').str.strip()
    # print(specs_df.dtypes)
    for col in specs_df.columns:
        assert col in allowed_columns, "Unrecognized column: " + col
    for index, row in specs_df.iterrows():
        run_single_trainer(row, models_dir, out_dir, err_dir)


def main():
    parser = argparse.ArgumentParser(description="Run multiple trainers with spec inputs from a csv file")
    parser.add_argument("--specs", required=False, default=None, type=str)
    args = parser.parse_args()
    run_trainers(args.specs)


if __name__ == '__main__':
    main()
