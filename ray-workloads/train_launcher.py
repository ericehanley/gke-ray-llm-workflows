'''
Notes on the train_launcher.py script:
This training workflow is launched via the custom train_launcher.py script instead of llamafactory-cli to ensure compatibility with cloud object storage like GCS.
The default training command creates a race condition when multiple workers attempt to save checkpoints simultaneously to a GCS Fuse mount.
This launcher uses the standard ray.train.torch.TorchTrainer and a custom callback to designate a single worker for writing checkpoints, preventing I/O errors while maintaining compatibility with Ray Train's synchronization protocol.

Additionally, there is currently a BUG in this setup where llamafactory shuts down the PyTorch distributed communication service, and then Ray attempts to as well resulting in an AssertionError (i.e., assert pg is not None)
 - hence the benign shutdown handling in the main script.

In hindsight, it may have been better to develop our own demo from scratch. :P
'''

import os
import sys
import dataclasses
from inspect import signature
from typing import List, Dict

from ray import train # Import ray.train and required 
from ray.train import ScalingConfig, RunConfig
from ray.train.base_trainer import TrainingFailedError
from ray.train.torch import TorchTrainer

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
    '''
    This callback functions as a workaround to prevent an error from a race condition
    when using Llama Factory. See note above.
    '''
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
    '''
    Train_loop_per_worker is the fundamental pattern for ray.train.
    This function executes on each Ray worker. run_exp is the LlamaFactory
    function that enables distributed training.
    '''
    callbacks = config.pop("custom_callbacks", [])
    run_exp(args=config, callbacks=callbacks)


def main():

    # LLamaFactory handling and configuration
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
    ray_num_workers = filtered_args.pop("ray_num_workers", 1)
    resources_per_worker = filtered_args.pop("resources_per_worker", {"GPU": 1})
    ray_storage_path = filtered_args.pop("ray_storage_path")
    ray_run_name = filtered_args.pop("ray_run_name")

    # Scaling_config is ray train paradigm - defines #workers and resource
    # configuration per worker.
    scaling_config = ScalingConfig(
        num_workers=ray_num_workers,
        resources_per_worker=resources_per_worker,
        use_gpu=True
    )
    # Run_config defines storage path and logging settings.
    run_config = RunConfig(storage_path=ray_storage_path,
        name=ray_run_name
        )

    train_loop_config = filtered_args
    train_loop_config["custom_callbacks"] = [RankZeroReportCallback()]

    '''
    A Trainer for data parallel PyTorch training.
    At a high level, this Trainer does the following:
    Launches multiple workers as defined by the scaling_config.
    Sets up a distributed PyTorch environment on these workers as defined by the torch_config.
    Ingests the input datasets based on the dataset_config.
    Runs the input train_loop_per_worker(train_loop_config) on all workers.
    '''
    trainer = TorchTrainer(
        train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    try:
        result = trainer.fit() # Standard approach to calling trainer object and kicking off distributed training
        print("--- Training Job SUCCEEDED ---")
        print(f"Final result: {result}")

    except TrainingFailedError as e:
        # Recursively search the exception chain for our specific benign error.
        # Nothing to see here please move along sir...
        is_benign_shutdown_error = False
        current_exception = e
        while current_exception:
            if "assert pg is not None" in str(current_exception):
                is_benign_shutdown_error = True
                break
            current_exception = current_exception.__cause__

        if is_benign_shutdown_error:
            print("--- Training Job SUCCEEDED ---")
            print("Successfully completed training and checkpointing.")
            print("Caught and ignored the known, benign shutdown assertion error.")
        else:
            # This was a real, unexpected error. Re-raise it.
            print(f"Caught an unexpected and critical error during training:")
            raise e

if __name__ == "__main__":
    main()