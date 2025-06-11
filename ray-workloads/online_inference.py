import os
import argparse
from openai import OpenAI  # to use openai api format
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

model_id = "NER_FT_QWEN"
model_source = "Qwen/Qwen2.5-7B-Instruct"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Deploy a Ray Serve LLM with a dynamic LoRA path.")
    
    parser.add_argument(
        "--dynamic-lora-path",
        type=str,
        required=True,
        help='The GCS URI (e.g., "gs://<bucket>/path/to/lora/") for dynamic LoRA loading.'
    )

    args = parser.parse_args()

    # Define config.
    llm_config = LLMConfig(
        model_loading_config={
            "model_id": model_id,
            "model_source": model_source
        },
        lora_config={  # REMOVE this section if you're only using a base model.
            "dynamic_lora_loading_path": args.dynamic_lora_path,
            "max_num_adapters_per_replica": 16,  # You only have 1.
        },
        # runtime_env={"env_vars": {"HF_TOKEN": os.environ.get("HF_TOKEN")}},
        deployment_config={
            "autoscaling_config": {
                "min_replicas": 1,
                "max_replicas": 2,
                # complete list: https://docs.ray.io/en/latest/serve/autoscaling-guide.html#serve-autoscaling
            }
        },
        accelerator_type="L4",
        engine_kwargs={
            "max_model_len": 4096,  # Or increase KV cache size.
            "tensor_parallel_size": 1,
            "enable_lora": True,
            # complete list: https://docs.vllm.ai/en/stable/serving/engine_args.html
        },
    )

    # Deploy.
    app = build_openai_app({"llm_configs": [llm_config]})
    serve.run(app)