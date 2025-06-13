import os
import argparse
from openai import OpenAI
from ray import serve # Using ray.serve to serve!
from ray.serve.llm import LLMConfig, build_openai_app

model_id = "NER_FT_QWEN"
model_source = "Qwen/Qwen2.5-7B-Instruct"

if __name__ == "__main__":

    # Extract dynamic-lora-path! Must pass as argument.
    parser = argparse.ArgumentParser(description="Deploy a Ray Serve LLM with a dynamic LoRA path.")
    parser.add_argument(
        "--dynamic-lora-path",
        type=str,
        required=True,
        help='The GCS URI (e.g., "gs://<bucket>/path/to/lora/") for dynamic LoRA loading.'
    )

    args = parser.parse_args()

    # Define configuration for vLLM server!
    llm_config = LLMConfig(
        model_loading_config={
            "model_id": model_id,
            "model_source": model_source
        },
        lora_config={
            "dynamic_lora_loading_path": args.dynamic_lora_path,
            "max_num_adapters_per_replica": 16,
        },
        deployment_config={
            "autoscaling_config": {
                "min_replicas": 1,
                "max_replicas": 2,
            }
        },
        accelerator_type="L4",
        engine_kwargs={
            "max_model_len": 4096,
            "tensor_parallel_size": 1,
            "enable_lora": True,
        },
    )

    # Deploy.
    app = build_openai_app({"llm_configs": [llm_config]})
    serve.run(app)