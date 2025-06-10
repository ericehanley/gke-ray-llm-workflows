# launcher.py
import os
import sys
import dataclasses
from inspect import signature
from typing import List, Dict

# Ray imports
from ray import train
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer

# PyTorch import for the fix
import torch
import torch.distributed as dist

# LlamaFactory imports
from transformers import TrainerCallback, HfArgumentParser
from llamafactory.train.tuner import run_exp
from llamafactory.hparams import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    FinetuningArguments,
    GeneratingArguments,
)


class RankZeroReportCallback(TrainerCallback):
    """
    This callback satisfies two requirements:
    1.  Only Rank 0 writes the checkpoint data to the shared filesystem (GCS).
    2.  All workers call `train.report()` to stay in sync with Ray Train.
    """
    def on_save(self, args, state, control, **kwargs):
        metrics = {k: v for k, v in state.log_history[-1].items() if isinstance(v, (int, float))}
        
        if train.get_context().get_world_rank() == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.isdir(checkpoint_path):
                train.report(metrics=metrics, checkpoint=train.Checkpoint.from_directory(checkpoint_path))
            else:
                train.report(metrics=metrics)
        else:
            train.report(metrics=metrics)


def train_loop_per_worker(config: Dict):
    """
    This is the function that Ray Train will execute on each of the distributed workers.
    """
    callbacks = config.pop("custom_callbacks", [])
    run_exp(args=config, callbacks=callbacks)
    


def main():
    # 1. Define and parse arguments from the YAML file
    dataclass_types = (
        ModelArguments, DataArguments, TrainingArguments, FinetuningArguments, GeneratingArguments
    )
    parser = HfArgumentParser(dataclass_types)
    if not (len(sys.argv) == 2 and sys.argv[1].endswith(".yaml")):
        print("Error: Please provide the path to your YAML configuration file.")
        sys.exit(1)
    
    args_tuple = parser.parse_yaml_file(os.path.abspath(sys.argv[1]))
    
    all_args = {}
    for arg_obj in args_tuple:
        all_args.update(dataclasses.asdict(arg_obj))

    known_keys = {key for dc_type in dataclass_types for key in signature(dc_type).parameters.keys()}
    filtered_args = {k: v for k, v in all_args.items() if k in known_keys}

    # 2. Extract Ray-specific configs
    ray_num_workers = filtered_args.pop("ray_num_workers", 1)
    resources_per_worker = filtered_args.pop("resources_per_worker", {"GPU": 1})
    ray_storage_path = filtered_args.pop("ray_storage_path")
    ray_run_name = filtered_args.pop("ray_run_name")

    # 3. Configure Ray Train Scaling and Run configs
    scaling_config = ScalingConfig(
        num_workers=ray_num_workers,
        resources_per_worker=resources_per_worker,
        use_gpu=True
    )
    run_config = RunConfig(storage_path=ray_storage_path, name=ray_run_name)

    # 4. Define the config to be passed to each worker
    train_loop_config = filtered_args
    train_loop_config["custom_callbacks"] = [RankZeroReportCallback()]

    # 5. Instantiate and run the Ray TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        run_config=run_config,
    )
    result = trainer.fit()

    print("--- Training Job Finished ---")
    print(f"Final result: {result}")


if __name__ == "__main__":
    main()