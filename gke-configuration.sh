export REGION=us-west1
export ZONE=us-west1-a
export PROJECT_ID=
export GKE_VERSION=1.32.2-gke.1297002
export CLUSTER_NAME=
export GSBUCKET=
export ARTIFACTREPO=
export PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
export NAMESPACE=default
export KSA_NAME=
export NUM_NODES=1
export NUM_GPUS_PER_NODE=4

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

# Deploy RayCluster - UPDATE CONTAINER IMAGE, BUCKET AND KSA MANUALLY
kubectl apply -f configs/raycluster-deploy.yaml

# Leverage virtual environment and install ray API locally to run jobs
python -m venv myenv #update apt as needed
source myenv/bin/activate
pip install -U "ray[data,train,tune,serve]"

# Submit ray jobs to cluster
# Get the head pod name
export HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head,ray.io/cluster=raycluster-demo -o jsonpath='{.items[0].metadata.name}')
echo "Head pod: $HEAD_POD"

# Port-forward (keep this running in a separate terminal)
kubectl port-forward $HEAD_POD 8265:8265

# In separate terminal, activate venv and submit job to RayCluster
cd gke-ray-llm-workflows
source myenv/bin/activate

# Submit data ingestion job
ray job submit --address http://localhost:8265 --working_dir: "." -- python ray-workloads/ingest_data.py

# Submit training job
ray job submit --address http://localhost:8265 --working-dir="." -- python ray-workloads/train_launcher.py ray-workloads/lora_sft_ray.yaml

# Submit batch inference job
ray job submit --address http://localhost:8265 --working-dir="." -- python ray-workloads/batch_inference.py

# Submit online inference
ray job submit --address http://localhost:8265 --working-dir="." -- python ray-workloads/online_inference.py

# Submit requests to online inference service
export HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head,ray.io/cluster=raycluster-demo -o jsonpath='{.items[0].metadata.name}')
kubectl port-forward $HEAD_POD 8000:8000

# In ANOTHER terminal
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "NER_FT_QWEN:checkpoint-93",
    "messages": [
      {
        "role": "user",
        "content": "Extract the person and location from this sentence: Maria traveled from Berlin to meet her friend."
      }
    ],
    "temperature": 0.2
  }'