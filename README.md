# Entity Recognition with LLMs - Ray on GKE Fork

This end-to-end tutorial **fine-tunes** an LLM to perform **batch inference** and **online serving** at scale. While entity recognition (NER) is the main task in this tutorial, you can easily extend these end-to-end workflows to any use case.

**This fork has been updated to run specifically on GKE and highlights the functionality of KubeRay CRDs.**

<img src="https://raw.githubusercontent.com/anyscale/e2e-llm-workflows/refs/heads/main/images/e2e_llm.png" width=800>

**Note**: The intent of this tutorial is to show how you can use Ray to implement end-to-end LLM workflows that can extend to any use case, including multimodal.

This tutorial uses the [Ray library](https://github.com/ray-project/ray) to implement these workflows, namely the LLM APIs:

[`ray.data.llm`](https://docs.ray.io/en/latest/data/working-with-llms.html):
- Batch inference over distributed datasets
- Streaming and async execution for throughput
- Built-in metrics and tracing, including observability
- Zero-copy GPU data transfer
- Composable with preprocessing and postprocessing steps

[`ray.serve.llm`](https://docs.ray.io/en/latest/serve/llm/serving-llms.html):
- Automatic scaling and load balancing
- Unified multi-node multi-model deployment
- Multi-LoRA support with shared base models
- Deep integration with inference engines, vLLM to start
- Composable multi-model LLM pipelines

And all of these workloads come with all the observability views you need to debug and tune them to **maximize throughput/latency**.

## Set up

### Cloud Environment Set Up
The infrastructure deployment can now be found in configs at **gke-configuration.sh** and is described below.
The set up assumes you are in an authenticated CLI with requisite permissions. Working from Google Cloud Shell is the easiest way to ensure all required tools are accessible.

Start by cloning this repo to your working environment

```bash
git clone https://github.com/ericehanley/gke-ray-llm-workflows.git
```

#### Set Environment Variables

```bash
export REGION=us-west1
export ZONE=us-west1-a
export PROJECT_ID= #Enter Project_ID
export GKE_VERSION=1.32.2-gke.1297002
export CLUSTER_NAME= #Enter Cluster Name
export GSBUCKET= #Enter Bucket Name
export ARTIFACTREPO= #Enter Artifact Registry Repo Name
export PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
export NAMESPACE=default
export KSA_NAME= #Enter Kubernetes Service Account Name
export NUM_NODES=1
export NUM_GPUS_PER_NODE=4
```

#### Create GKE Cluster and L4 Node Pool

Note in the GKE Cluster create command, our RayOperator addon automatically handles the installation of the KubeRay operator.


```bash
gcloud container clusters create ${CLUSTER_NAME} \
    --region=${REGION} \
    --node-locations=${ZONE} \
    --cluster-version=${GKE_VERSION} \
    --machine-type=n2-standard-8 \
    --num-nodes=1 \
    --enable-ray-cluster-logging \
    --enable-ray-cluster-monitoring \
    --workload-pool=${PROJECT_ID}.svc.id.goog \
    --addons=RayOperator,GcsFuseCsiDriver

gcloud container node-pools create l4singlenodepool \
    --accelerator type=nvidia-l4,count=4,gpu-driver-version=latest \
    --node-version=${GKE_VERSION} \
    --project=${PROJECT_ID} \
    --region=${REGION} \
    --node-locations=${ZONE} \
    --cluster=${CLUSTER_NAME} \
    --machine-type=g2-standard-48 \
    --num-nodes=${NUM_NODES} \
    --disk-size=200GB
```

#### Create GCS Bucket and Configure Service Account for Bucket Access

```bash
# Create Cloud Storage Bucket & Configure
gcloud storage buckets create gs://${GSBUCKET} \
    --uniform-bucket-level-access \
    --location=${REGION}\
    --enable-hierarchical-namespace

# Create GKE SA
kubectl create serviceaccount ${KSA_NAME}

# Add permissions to bucket for FUSE CSI driver
gcloud storage buckets add-iam-policy-binding gs://${GSBUCKET} \
  --member "principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/${NAMESPACE}/sa/${KSA_NAME}" \
  --role "roles/storage.objectUser"
```

#### Dependencies
The Dockerfile and requirements.txt can be used to reliably build an image that works for our demo.

We can use **Cloud Build** and **Artifact Registry** to build and store our container images.

```bash
# Create repository and image
gcloud artifacts repositories ${ARTIFACTREPO} \
    --repository-format=docker \
    --location=${REGION} \
    --description="Docker repository for Ray applications"

export CLOUD_BUILD_SA="service-${PROJECT_NUMBER}@gcp-sa-cloudbuild.iam.gserviceaccount.com"

gcloud artifacts repositories add-iam-policy-binding ${ARTIFACTREPO} \
    --location=${REGION} \
    --member="serviceAccount:${CLOUD_BUILD_SA}" \
    --role="roles/artifactregistry.writer"

export IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/ray-docker-repo/gke-llama-factory:v1"

gcloud builds submit container-image/ --tag $IMAGE_NAME
```

#### RayCluster Deployment
Now we can use the *raycluster-deploy.yaml* file to deploy our RayCluster CRD to our GKE Cluster.

**NOTE: You MUST manually update the bucket name, kubernetes service account name, and container image name in the raycluster-deploy.yaml file prior to submitting the manifest.**

```bash
kubectl apply -f configs/raycluster-deploy.yaml
```
#### Leverage Virtual Environment for Ray API Calls
Now we can activate a virtual environment in our terminal to effectively leverage the ray API.

```bash
python -m venv myenv
source myenv/bin/activate
pip install -U "ray[data,train,tune,serve]"
```

Finally, we can port-forward to our RayCluster to be able to submit jobs in a separate terminal.

**In current terminal:**
```bash
export HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head,ray.io/cluster=raycluster-demo -o jsonpath='{.items[0].metadata.name}')
kubectl port-forward $HEAD_POD 8265:8265
```

**In new terminal (reactivate virtual environment):**
```bash
cd gke-ray-llm-workflows
source myenv/bin/activate
```

## Data Ingestion

The data ingestion process has been revised in this fork to run as a ray job submitted to our cluster. The job is defined in *ray-workloads/ingest_data.py*:

```python
import subprocess
import os
import time

# The shared path where the GCS bucket is mounted
VIGGO_PATH = "/mnt/cluster_storage/viggo"
DATASET_INFO_FILE = os.path.join(VIGGO_PATH, "dataset_info.json")

def run_command(command):
    """Runs a shell command and raises an exception if it fails."""
    print(f"Executing: {' '.join(command)}")
    subprocess.run(" ".join(command), shell=True, check=True)

print("Starting data setup job...")

if os.path.exists(DATASET_INFO_FILE):
    print(f"Data already exists at {VIGGO_PATH}. Setup is complete.")
    exit(0)

print(f"Data not found. Starting download to {VIGGO_PATH}...")
run_command(["mkdir", "-p", VIGGO_PATH])

# --- Download all the required files ---
urls = {
    "train.jsonl": "https://viggo-ds.s3.amazonaws.com/train.jsonl",
    "val.jsonl": "https://viggo-ds.s3.amazonaws.com/val.jsonl",
    "test.jsonl": "https://viggo-ds.s3.amazonaws.com/test.jsonl",
    "dataset_info.json": "https://viggo-ds.s3.amazonaws.com/dataset_info.json"
}

for filename, url in urls.items():
    output_path = os.path.join(VIGGO_PATH, filename)
    run_command(["wget", url, "-O", output_path])
    time.sleep(1) # Small delay to be courteous to the server

print("\nAll files downloaded successfully.")
print("Data setup job finished.")
```

We download the files to a the mounted location on our cluster - they will now appear in our bucket.

Below is a single example of the structure of our training data: a JSON file with an instruction field, an input field, and an output field.

    {
        "instruction": "Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values. This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute']. The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']",
        "input": "Blizzard North is mostly an okay developer, but they released Diablo II for the Mac and so that pushes the game from okay to good in my view.",
        "output": "give_opinion(name[Diablo II], developer[Blizzard North], rating[good], has_mac_release[yes])"
    }

### Submitting the Data Ingestion Job
In our new terminal from the set up above, we can now submit our jobs with a simple one line command:

```bash
ray job submit --address http://localhost:8265 --working_dir: "." -- python ray-workloads/ingest_data.py
```

## Distributed fine-tuning

Use [Ray Train](https://docs.ray.io/en/latest/train/train.html) + [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to perform multi-node training. Find the parameters for the training workload, post-training method, dataset location, train/val details, etc. in the `llama3_lora_sft_ray.yaml` config file. See the recipes for even more post-training methods, like SFT, pretraining, PPO, DPO, KTO, etc. [on GitHub](https://github.com/hiyouga/LLaMA-Factory/tree/main/examples).

**Note**: Ray also supports using other tools like [axolotl](https://axolotl-ai-cloud.github.io/axolotl/docs/ray-integration.html) or even [Ray Train + HF Accelerate + FSDP/DeepSpeed](https://docs.ray.io/en/latest/train/huggingface-accelerate.html) directly for complete control of your post-training workloads.

<img src="https://raw.githubusercontent.com/anyscale/foundational-ray-app/refs/heads/main/images/distributed_training.png" width=800>

### lora_sft_ray.yaml

Below is an overview of the configuration for our model. We specify:
 * The model name and 
 * The method by which we will be tuning (LORA)
 * The training dataset along with configuration on how the data should be processed.
 * Output location for logs and checkpoints.
 * Ray-specific configuration
 * Training hypterparamters
 * Evaluation dataset configuration

```yaml
### model
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: viggo-train
dataset_dir: /mnt/cluster_storage/viggo  # shared storage workers have access to
template: qwen
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: /mnt/cluster_storage/viggo/outputs  # should be somewhere workers have access to (ex. s3, nfs)
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false

### ray
ray_run_name: lora_sft_ray
ray_storage_path: /mnt/cluster_storage/ray_results
ray_num_workers: 4
resources_per_worker:
  GPU: 1
placement_strategy: PACK

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### eval
eval_dataset: viggo-val  # uses same dataset_dir as training data
# val_size: 0.1  # only if using part of training data for validation
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
```

### Multi-node training

Use Ray Train + LlamaFactory to perform the multi-node train loop.

<div class="alert alert-block alert"> <b>Ray Train</b>

Using [Ray Train](https://docs.ray.io/en/latest/train/train.html) has several advantages:
- it automatically handles **multi-node, multi-GPU** setup with no manual SSH setup or `hostfile` configs.
- you can define **per-worker fractional resource requirements**, for example, 2 CPUs and 0.5 GPU per worker.
- you can run on **heterogeneous machines** and scale flexibly, for example, CPU for preprocessing and GPU for training.
- it has built-in **fault tolerance** through retry of failed workers, and continue from last checkpoint.
- it supports Data Parallel, Model Parallel, Parameter Server, and even custom strategies.
- [Ray Compiled graphs](https://docs.ray.io/en/latest/ray-core/compiled-graph/ray-compiled-graph.html) allow you to even define different parallelism for jointly optimizing multiple models. Megatron, DeepSpeed, and similar frameworks only allow for one global setting.

Because our RayCluster is already up and running, we can leverage the ray API to submit a job to the RayCluster just like we did for the data ingestion workload.

```bash
ray job submit --address http://localhost:8265 --working-dir="." -- bash -c "USE_RAY=1 llamafactory-cli train ray-workloads/lora_sft_ray.yaml"
```
**Note on the train_launcher.py script:**
This training workflow is launched via the custom train_launcher.py script instead of *llamafactory-cli* to ensure compatibility with cloud object storage like GCS. The default training command creates a race condition when multiple workers attempt to save checkpoints simultaneously to a GCS Fuse mount. This launcher uses the standard ray.train.torch.TorchTrainer and a custom callback to designate a single worker for writing checkpoints, preventing I/O errors while maintaining compatibility with Ray Train's synchronization protocol.


    Training started with configuration:
        ╭──────────────────────────────────────────────────────────────────────────────────────────────────────╮
        │ Training config                                                                                      │
        ├──────────────────────────────────────────────────────────────────────────────────────────────────────┤
        │ train_loop_config/args/bf16                                                                     True │
        │ train_loop_config/args/cutoff_len                                                               2048 │
        │ train_loop_config/args/dataloader_num_workers                                                      4 │
        │ train_loop_config/args/dataset                                                           viggo-train │
        │ train_loop_config/args/dataset_dir                                              ...ter_storage/viggo │
        │ train_loop_config/args/ddp_timeout                                                         180000000 │
        │ train_loop_config/args/do_train                                                                 True │
        │ train_loop_config/args/eval_dataset                                                        viggo-val │
        │ train_loop_config/args/eval_steps                                                                500 │
        │ train_loop_config/args/eval_strategy                                                           steps │
        │ train_loop_config/args/finetuning_type                                                          lora │
        │ train_loop_config/args/gradient_accumulation_steps                                                 8 │
        │ train_loop_config/args/learning_rate                                                          0.0001 │
        │ train_loop_config/args/logging_steps                                                              10 │
        │ train_loop_config/args/lora_rank                                                                   8 │
        │ train_loop_config/args/lora_target                                                               all │
        │ train_loop_config/args/lr_scheduler_type                                                      cosine │
        │ train_loop_config/args/max_samples                                                              1000 │
        │ train_loop_config/args/model_name_or_path                                       ...en2.5-7B-Instruct │
        │ train_loop_config/args/num_train_epochs                                                          5.0 │
        │ train_loop_config/args/output_dir                                               ...age/viggo/outputs │
        │ train_loop_config/args/overwrite_cache                                                          True │
        │ train_loop_config/args/overwrite_output_dir                                                     True │
        │ train_loop_config/args/per_device_eval_batch_size                                                  1 │
        │ train_loop_config/args/per_device_train_batch_size                                                 1 │
        │ train_loop_config/args/placement_strategy                                                       PACK │
        │ train_loop_config/args/plot_loss                                                                True │
        │ train_loop_config/args/preprocessing_num_workers                                                  16 │
        │ train_loop_config/args/ray_num_workers                                                             4 │
        │ train_loop_config/args/ray_run_name                                                     lora_sft_ray │
        │ train_loop_config/args/ray_storage_path                                         ...orage/viggo/saves │
        │ train_loop_config/args/resources_per_worker/GPU                                                    1 │
        │ train_loop_config/args/resources_per_worker/anyscale/accelerator_shape:4xL4                      1 │
        │ train_loop_config/args/resume_from_checkpoint                                                        │
        │ train_loop_config/args/save_only_model                                                         False │
        │ train_loop_config/args/save_steps                                                                500 │
        │ train_loop_config/args/stage                                                                     sft │
        │ train_loop_config/args/template                                                                 qwen │
        │ train_loop_config/args/trust_remote_code                                                        True │
        │ train_loop_config/args/warmup_ratio                                                              0.1 │
        │ train_loop_config/callbacks                                                     ... 0x7e1262910e10>] │
        ╰──────────────────────────────────────────────────────────────────────────────────────────────────────╯

        100%|██████████| 155/155 [07:12<00:00,  2.85s/it][INFO|trainer.py:3942] 2025-04-11 14:57:59,207 >> Saving model checkpoint to /mnt/cluster_storage/viggo/outputs/checkpoint-155

        Training finished iteration 1 at 2025-04-11 14:58:02. Total running time: 10min 24s
        ╭─────────────────────────────────────────╮
        │ Training result                         │
        ├─────────────────────────────────────────┤
        │ checkpoint_dir_name   checkpoint_000000 │
        │ time_this_iter_s              521.83827 │
        │ time_total_s                  521.83827 │
        │ training_iteration                    1 │
        │ epoch                             4.704 │
        │ grad_norm                       0.14288 │
        │ learning_rate                        0. │
        │ loss                             0.0065 │
        │ step                                150 │
        ╰─────────────────────────────────────────╯
        Training saved a checkpoint for iteration 1 at: (local)/mnt/cluster_storage/viggo/saves/lora_sft_ray/TorchTrainer_95d16_00000_0_2025-04-11_14-47-37/checkpoint_000000


```python
display(Code(filename="/mnt/cluster_storage/viggo/outputs/all_results.json", language="json"))
```
<div class="highlight"><pre><span></span><span class="p">{</span>
<span class="w">    </span><span class="nt">&quot;epoch&quot;</span><span class="p">:</span><span class="w"> </span><span class="mf">4.864</span><span class="p">,</span>
<span class="w">    </span><span class="nt">&quot;eval_viggo-val_loss&quot;</span><span class="p">:</span><span class="w"> </span><span class="mf">0.13618840277194977</span><span class="p">,</span>
<span class="w">    </span><span class="nt">&quot;eval_viggo-val_runtime&quot;</span><span class="p">:</span><span class="w"> </span><span class="mf">20.2797</span><span class="p">,</span>
<span class="w">    </span><span class="nt">&quot;eval_viggo-val_samples_per_second&quot;</span><span class="p">:</span><span class="w"> </span><span class="mf">35.208</span><span class="p">,</span>
<span class="w">    </span><span class="nt">&quot;eval_viggo-val_steps_per_second&quot;</span><span class="p">:</span><span class="w"> </span><span class="mf">8.827</span><span class="p">,</span>
<span class="w">    </span><span class="nt">&quot;total_flos&quot;</span><span class="p">:</span><span class="w"> </span><span class="mf">4.843098686147789e+16</span><span class="p">,</span>
<span class="w">    </span><span class="nt">&quot;train_loss&quot;</span><span class="p">:</span><span class="w"> </span><span class="mf">0.2079355036479331</span><span class="p">,</span>
<span class="w">    </span><span class="nt">&quot;train_runtime&quot;</span><span class="p">:</span><span class="w"> </span><span class="mf">437.2951</span><span class="p">,</span>
<span class="w">    </span><span class="nt">&quot;train_samples_per_second&quot;</span><span class="p">:</span><span class="w"> </span><span class="mf">11.434</span><span class="p">,</span>
<span class="w">    </span><span class="nt">&quot;train_steps_per_second&quot;</span><span class="p">:</span><span class="w"> </span><span class="mf">0.354</span>
<span class="p">}</span>
</pre></div>



<img src="https://raw.githubusercontent.com/anyscale/e2e-llm-workflows/refs/heads/main/images/loss.png" width=500>

## Batch inference
[`Overview`](https://docs.ray.io/en/latest/data/working-with-llms.html) |  [`API reference`](https://docs.ray.io/en/latest/data/api/llm.html)

The `ray.data.llm` module integrates with key large language model (LLM) inference engines and deployed models to enable LLM batch inference. These LLM modules use [Ray Data](https://docs.ray.io/en/latest/data/data.html) under the hood, which makes it extremely easy to distribute workloads but also ensures that they happen:
- **efficiently**: minimizing CPU/GPU idle time with heterogeneous resource scheduling.
- **at scale**: with streaming execution to petabyte-scale datasets, especially when [working with LLMs](https://docs.ray.io/en/latest/data/working-with-llms.html).
- **reliably** by checkpointing processes, especially when running workloads on spot instances with on-demand fallback.
- **flexibly**: connecting to data from any source, applying transformations, and saving to any format and location for your next workload.

<img src="https://raw.githubusercontent.com/anyscale/foundational-ray-app/refs/heads/main/images/ray_data_solution.png" width=800>

[RayTurbo Data](https://docs.anyscale.com/rayturbo/rayturbo-data) has more features on top of Ray Data:
- **accelerated metadata fetching** to improve reading first time from large datasets
- **optimized autoscaling** where Jobs can kick off before waiting for the entire cluster to start
- **high reliability** where entire failed jobs, like head node, cluster, uncaptured exceptions, etc., can resume from checkpoints. OSS Ray can only recover from worker node failures.

Start by defining the [vLLM engine processor config](https://docs.ray.io/en/latest/data/api/doc/ray.data.llm.vLLMEngineProcessorConfig.html#ray.data.llm.vLLMEngineProcessorConfig) where you can select the model to use and the [engine behavior](https://docs.vllm.ai/en/stable/serving/engine_args.html). The model can come from [Hugging Face (HF) Hub](https://huggingface.co/models) or a local model path `/path/to/your/model`. Anyscale supports GPTQ, GGUF, or LoRA model formats.

<img src="https://raw.githubusercontent.com/anyscale/e2e-llm-workflows/refs/heads/main/images/data_llm.png" width=800>

### vLLM engine processor


```python
import os
import ray
from ray.data.llm import vLLMEngineProcessorConfig
```

    INFO 04-11 14:58:40 __init__.py:194] No platform detected, vLLM is running on UnspecifiedPlatform



```python
config = vLLMEngineProcessorConfig(
    model_source=model_source,
    runtime_env={
        "env_vars": {
            "VLLM_USE_V1": "0",  # v1 doesn't support lora adapters yet
            # "HF_TOKEN": os.environ.get("HF_TOKEN"),
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
```

### LLM processor

Next, pass the config to an [LLM processor](https://docs.ray.io/en/master/data/api/doc/ray.data.llm.build_llm_processor.html#ray.data.llm.build_llm_processor) where you can define the preprocessing and postprocessing steps around inference. With your base model defined in the processor config, you can define the LoRA adapter layers as part of the preprocessing step of the LLM processor itself.


```python
from ray.data.llm import build_llm_processor
```


```python
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
```

    2025-04-11 14:58:40,942	INFO worker.py:1660 -- Connecting to existing Ray cluster at address: 10.0.51.51:6379...
    2025-04-11 14:58:40,953	INFO worker.py:1843 -- Connected to Ray cluster. View the dashboard at [1m[32mhttps://session-zt5t77xa58pyp3uy28glg2g24d.i.anyscaleuserdata.com [39m[22m
    2025-04-11 14:58:40,960	INFO packaging.py:367 -- Pushing file package 'gcs://_ray_pkg_e71d58b4dc01d065456a9fc0325ee2682e13de88.zip' (2.16MiB) to Ray cluster...
    2025-04-11 14:58:40,969	INFO packaging.py:380 -- Successfully pushed file package 'gcs://_ray_pkg_e71d58b4dc01d065456a9fc0325ee2682e13de88.zip'.



    config.json:   0%|          | 0.00/663 [00:00<?, ?B/s]


    [36m(pid=51260)[0m INFO 04-11 14:58:47 __init__.py:194] No platform detected, vLLM is running on UnspecifiedPlatform



```python
# Evaluation on test dataset
ds = ray.data.read_json("/mnt/cluster_storage/viggo/test.jsonl")  # complete list: https://docs.ray.io/en/latest/data/api/input_output.html
ds = processor(ds)
results = ds.take_all()
results[0]
```



    {
      "batch_uuid": "d7a6b5341cbf4986bb7506ff277cc9cf",
      "embeddings": null,
      "generated_text": "request(esrb)",
      "generated_tokens": [2035, 50236, 10681, 8, 151645],
      "input": "Do you have a favorite ESRB content rating?",
      "instruction": "Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values. This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute']. The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']",
      "messages": [
        {
          "content": "Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values. This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute']. The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']",
          "role": "system"
        },
        {
          "content": "Do you have a favorite ESRB content rating?",
          "role": "user"
        }
      ],
      "metrics": {
        "arrival_time": 1744408857.148983,
        "finished_time": 1744408863.09091,
        "first_scheduled_time": 1744408859.130259,
        "first_token_time": 1744408862.7087252,
        "last_token_time": 1744408863.089174,
        "model_execute_time": null,
        "model_forward_time": null,
        "scheduler_time": 0.04162892400017881,
        "time_in_queue": 1.981276035308838
      },
      "model": "/mnt/cluster_storage/viggo/saves/lora_sft_ray/TorchTrainer_95d16_00000_0_2025-04-11_14-47-37/checkpoint_000000/checkpoint",
      "num_generated_tokens": 5,
      "num_input_tokens": 164,
      "output": "request_attribute(esrb[])",
      "params": "SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.3, top_p=1.0, top_k=-1, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=250, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None)",
      "prompt": "<|im_start|>system
    Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values. This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute']. The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']<|im_end|>
    <|im_start|>user
    Do you have a favorite ESRB content rating?<|im_end|>
    <|im_start|>assistant
    ",
      "prompt_token_ids": [151644, "...", 198],
      "request_id": 94,
      "time_taken_llm": 6.028705836999961,
      "generated_output": "request(esrb)"
    }





```python
# Exact match (strict!)
matches = 0
for item in results:
    if item["output"] == item["generated_output"]:
        matches += 1
matches / float(len(results))
```




    0.6879039704524469



**Note**: The objective of fine-tuning here isn't to create the most performant model but to show that you can leverage it for downstream workloads, like batch inference and online serving at scale. However, you can increase `num_train_epochs` if you want to.

Observe the individual steps in the batch inference workload through the Anyscale Ray Data dashboard:

<img src="https://raw.githubusercontent.com/anyscale/e2e-llm-workflows/refs/heads/main/images/data_dashboard.png" width=1000>

<div class="alert alert-info">

💡 For more advanced guides on topics like optimized model loading, multi-LoRA, OpenAI-compatible endpoints, etc., see [more examples](https://docs.ray.io/en/latest/data/working-with-llms.html) and the [API reference](https://docs.ray.io/en/latest/data/api/llm.html).

</div>

## Online serving
[`Overview`](https://docs.ray.io/en/latest/serve/llm/serving-llms.html) | [`API reference`](https://docs.ray.io/en/latest/serve/api/index.html#llm-api)

<img src="https://raw.githubusercontent.com/anyscale/foundational-ray-app/refs/heads/main/images/ray_serve.png" width=600>

`ray.serve.llm` APIs allow users to deploy multiple LLM models together with a familiar Ray Serve API, while providing compatibility with the OpenAI API.

<img src="https://raw.githubusercontent.com/anyscale/e2e-llm-workflows/refs/heads/main/images/serve_llm.png" width=500>

Ray Serve LLM is designed with the following features:
- Automatic scaling and load balancing
- Unified multi-node multi-model deployment
- OpenAI compatibility
- Multi-LoRA support with shared base models
- Deep integration with inference engines, vLLM to start
- Composable multi-model LLM pipelines

[RayTurbo Serve](https://docs.anyscale.com/rayturbo/rayturbo-serve) on Anyscale has more features on top of Ray Serve:
- **fast autoscaling and model loading** to get services up and running even faster: [5x improvements](https://www.anyscale.com/blog/autoscale-large-ai-models-faster) even for LLMs
- 54% **higher QPS** and up-to 3x **streaming tokens per second** for high traffic serving use-cases
- **replica compaction** into fewer nodes where possible to reduce resource fragmentation and improve hardware utilization
- **zero-downtime** [incremental rollouts](https://docs.anyscale.com/platform/services/update-a-service/#resource-constrained-updates) so your service is never interrupted
- [**different environments**](https://docs.anyscale.com/platform/services/multi-app/#multiple-applications-in-different-containers) for each service in a multi-serve application
- **multi availability-zone** aware scheduling of Ray Serve replicas to provide higher redundancy to availability zone failures


### LLM serve config


```python
import os
from openai import OpenAI  # to use openai api format
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app
```

Define an [LLM config](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.llm.LLMConfig.html#ray.serve.llm.LLMConfig) where you can define where the model comes from, it's [autoscaling behavior](https://docs.ray.io/en/latest/serve/autoscaling-guide.html#serve-autoscaling), what hardware to use and [engine arguments](https://docs.vllm.ai/en/stable/serving/engine_args.html).


```python
# Define config.
llm_config = LLMConfig(
    model_loading_config={
        "model_id": model_id,
        "model_source": model_source
    },
    lora_config={  # REMOVE this section if you're only using a base model.
        "dynamic_lora_loading_path": dynamic_lora_path,
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
```

Now deploy the LLM config as an application. And because this application is all built on top of [Ray Serve](https://docs.ray.io/en/latest/serve/index.html), you can have advanced service logic around composing models together, deploying multiple applications, model multiplexing, observability, etc.


```python
# Deploy.
app = build_openai_app({"llm_configs": [llm_config]})
serve.run(app)
```

    DeploymentHandle(deployment='LLMRouter')


### Service request


```python
# Initialize client.
client = OpenAI(base_url="http://localhost:8000/v1", api_key="fake-key")
response = client.chat.completions.create(
    model=f"{model_id}:{lora_id}",
    messages=[
        {"role": "system", "content": "Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values. This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute']. The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']"},
        {"role": "user", "content": "Blizzard North is mostly an okay developer, but they released Diablo II for the Mac and so that pushes the game from okay to good in my view."},
    ],
    stream=True
)
for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```



    Avg prompt throughput: 20.3 tokens/s, Avg generation throughput: 0.1 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.3%, CPU KV cache usage: 0.0%.

    _opinion(name[Diablo II], developer[Blizzard North], rating[good], has_mac_release[yes])




And of course, you can observe the running service, the deployments, and metrics like QPS, latency, etc., through the [Ray Dashboard](https://docs.ray.io/en/latest/ray-observability/getting-started.html)'s [Serve view](https://docs.ray.io/en/latest/ray-observability/getting-started.html#dash-serve-view):

<img src="https://raw.githubusercontent.com/anyscale/e2e-llm-workflows/refs/heads/main/images/serve_dashboard.png" width=1000>

<div class="alert alert-info">

💡 See [more examples](https://docs.ray.io/en/latest/serve/llm/overview.html) and the [API reference](https://docs.ray.io/en/latest/serve/llm/api.html) for advanced guides on topics like structured outputs (like JSON), vision LMs, multi-LoRA on shared base models, using other inference engines (like `sglang`), fast model loading, etc.

</div>

```python
# Shutdown the service
serve.shutdown()
```

## Production

Seamlessly integrate with your existing CI/CD pipelines by leveraging the Anyscale [CLI](https://docs.anyscale.com/reference/quickstart-cli) or [SDK](https://docs.anyscale.com/reference/quickstart-sdk) to run [reliable batch jobs](https://docs.anyscale.com/platform/jobs) and deploy [highly available services](https://docs.anyscale.com/platform/services). Given you've been developing in an environment that's almost identical to production with a multi-node cluster, this integration should drastically speed up your dev to prod velocity.

<img src="https://raw.githubusercontent.com/anyscale/foundational-ray-app/refs/heads/main/images/cicd.png" width=600>

### Jobs

[Anyscale Jobs](https://docs.anyscale.com/platform/jobs/) ([API ref](https://docs.anyscale.com/reference/job-api/)) allows you to execute discrete workloads in production such as batch inference, embeddings generation, or model fine-tuning.
- [define and manage](https://docs.anyscale.com/platform/jobs/manage-jobs) your Jobs in many different ways, like CLI and Python SDK
- set up [queues](https://docs.anyscale.com/platform/jobs/job-queues) and [schedules](https://docs.anyscale.com/platform/jobs/schedules)
- set up all the [observability, alerting, etc.](https://docs.anyscale.com/platform/jobs/monitoring-and-debugging) around your Jobs

<img src="https://raw.githubusercontent.com/anyscale/foundational-ray-app/refs/heads/main/images/job_result.png" width=700>

### Services

[Anyscale Services](https://docs.anyscale.com/platform/services/) ([API ref](https://docs.anyscale.com/reference/service-api/)) offers an extremely fault tolerant, scalable, and optimized way to serve your Ray Serve applications:
- you can [rollout and update](https://docs.anyscale.com/platform/services/update-a-service) services with canary deployment with zero-downtime upgrades
- [monitor](https://docs.anyscale.com/platform/services/monitoring) your Services through a dedicated Service page, unified log viewer, tracing, set up alerts, etc.
- scale a service (`num_replicas=auto`) and utilize replica compaction to consolidate nodes that are fractionally utilized
- [head node fault tolerance](https://docs.anyscale.com/platform/services/production-best-practices#head-node-ft) because OSS Ray recovers from failed workers and replicas but not head node crashes
- serving [multiple applications](https://docs.anyscale.com/platform/services/multi-app) in a single Service

<img src="https://raw.githubusercontent.com/anyscale/foundational-ray-app/refs/heads/main/images/canary.png" width=700>



```bash
%%bash
# clean up
rm -rf /mnt/cluster_storage/viggo
STORAGE_PATH="$ANYSCALE_ARTIFACT_STORAGE/viggo"
if [[ "$STORAGE_PATH" == s3://* ]]; then
    aws s3 rm "$STORAGE_PATH" --recursive --quiet
elif [[ "$STORAGE_PATH" == gs://* ]]; then
    gsutil -m -q rm -r "$STORAGE_PATH"
fi
```
