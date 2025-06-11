import os
from pathlib import Path

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
            "max_model_len": 4096,  # or increase KV cache size
            # complete list: https://docs.vllm.ai/en/stable/serving/engine_args.html
        },
        concurrency=1,
        batch_size=16,
        accelerator_type="L4",
    )

    processor = build_llm_processor(
        config,
        preprocess=lambda row: dict(
            model=lora_path,  # REMOVE this line if doing inference with just the base model
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": row["input"]}
            ],
            sampling_params={
                "temperature": 0.3,
                "max_tokens": 250,
                # complete list: https://docs.vllm.ai/en/stable/api/inference_params.html
            },
        ),
        postprocess=lambda row: {
            **row,  # all contents
            "generated_output": row["generated_text"],
            # add additional outputs
        },
    )

    # Execute read/process/output
    ds = ray.data.read_json("/mnt/cluster_storage/viggo/test.jsonl")
    ds = processor(ds)
    results = ds.take_all()
    print(f"Example Output: {results[0]}")
    ds.write_json("/mnt/cluster_storage/viggo/batch_output.json")

if __name__ == "__main__":
    main()