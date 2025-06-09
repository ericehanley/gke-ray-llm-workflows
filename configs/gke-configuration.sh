export REGION=us-west1
export ZONE=us-west1-a
export PROJECT_ID=northam-ce-mlai-tpu
export GKE_VERSION=1.32.2-gke.1297002
export CLUSTER_NAME=ray-enabled-cluster
export GSBUCKET=eh-ray
export PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
export NAMESPACE=default
export KSA_NAME=eh-ray
export NUM_NODES=1
export NUM_GPUS_PER_NODE=4
export HF_TOKEN=

# Provision a single node, ray-enabled gke cluster
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
    --num-nodes=${NUM_NODES}



# Create a GCS bucket
gsutil mb gs://your-unique-bucket-name/

# Upload the data
gsutil cp \
  https://viggo-ds.s3.amazonaws.com/train.jsonl \
  https://viggo-ds.s3.amazonaws.com/val.jsonl \
  https://viggo-ds.s3.amazonaws.com/test.jsonl \
  https://viggo-ds.s3.amazonaws.com/dataset_info.json \
  gs://your-unique-bucket-name/viggo/


# Leverage virtual environment and install ray
python -m venv myenv && \
source myenv/bin/activate

pip install -U "ray[data,train,tune,serve]"

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

# Deploy RayCluster
kubectl create secret generic hf-secret --from-literal=HF_TOKEN=${HF_TOKEN}

# UPDATE ray-cluster-llama.yaml with SA and Bucket values and deploy
envsubst < a3-mega/ray-cluster-config.yaml | kubectl apply -f -

# Submit finetune job.
# Get the head pod name
export HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head,ray.io/cluster=llama-raycluster -o jsonpath='{.items[0].metadata.name}')
echo "Head pod: $HEAD_POD"

# Port-forward (keep this running in a separate terminal)
kubectl port-forward $HEAD_POD 8265:8265