import os
from pathlib import Path

# Import ray.data libraries
import ray
from ray.data.llm import vLLMEngineProcessorConfig
from ray.data.llm import build_llm_processor


system_content = ''' Given a target sentence construct the underlying meaning representation of the
input sentence as a single function with attributes and attribute values. This
function should describe the target string accurately and the function must be
one of the following ['inform', 'request', 'give_opinion', 'confirm',
'verify_attribute', 'suggest', 'request_explanation', 'recommend',
'request_attribute']. The attributes must be one of the following: ['name',
'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres',
'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam',
'has_linux_release', 'has_mac_release', 'specifier']'''

lora_path = "/mnt/cluster_storage/viggo/output/checkpoint-93/"

def main():
    # Configure vLLM settings that should be used for inference
    # on each actor
    config = vLLMEngineProcessorConfig(
        model_source="Qwen/Qwen2.5-7B-Instruct",
        runtime_env={
            "env_vars": {
                "VLLM_USE_V1": "0",
            },
        },
        engine_kwargs={
            "enable_lora": True,
            "max_lora_rank": 8,
            "max_loras": 1,
            "pipeline_parallel_size": 1,
            "tensor_parallel_size": 1,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 4096,
            "max_model_len": 4096,
        },
        concurrency=1,
        batch_size=16,
        accelerator_type="L4",
    )

    # Create the actual processor
    processor = build_llm_processor(
        config,
        preprocess=lambda row: dict(
            model=lora_path,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": row["input"]}
            ],
            sampling_params={
                "temperature": 0.3,
                "max_tokens": 250,
            },
        ),
        postprocess=lambda row: {
            **row,
            "generated_output": row["generated_text"],
        },
    )

    # Data sharding when reading inputs into smaller, blocks across machine
    ds = ray.data.read_json("/mnt/cluster_storage/viggo/test.jsonl")

    # Create a pool of Ray Actors on L4 accelerators
    # Each parallel actor has a dedicated vLLMEngine based on configuration
    # Efficiently stream the blocks to each of the actors
    ds = processor(ds)

    # Lazy execution - creates the distributed Dataset object - a blueprint that
    # is executed in parallel across the environment
    results = ds.take_all()


    print(f"Example Output: {results[0]}")

    # Write outputs in paralell too!
    ds.write_json("/mnt/cluster_storage/viggo/batch_output.json")

if __name__ == "__main__":
    main()