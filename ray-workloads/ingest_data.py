# Simple script to download viggo dataset to /mnt/cluster_storage/viggo
# which is mounted to GCS via GCSFUSEDRIVER
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