import os
import json
import time
import threading
import logging
import requests

from dotenv import load_dotenv
import runpod

from fastapi import FastAPI, Request
import uvicorn

from langchain_community.llms import HuggingFaceTextGenInference

# -------------------------------
# Logging setup
# -------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# -------------------------------
# Load .env and API key
# -------------------------------
load_dotenv()
api_key = os.getenv("RUNPOD_API_KEY")

if not api_key:
    logger.error("RUNPOD_API_KEY not found in .env file.")
    raise EnvironmentError("Missing RunPod API key")

runpod.api_key = api_key

# -------------------------------
# Display available GPUs
# -------------------------------
logger.info("Fetching available GPU types from RunPod:")
gpu_types = runpod.get_gpus()

id_width = 50
name_width = 20
vram_width = 12

header = (
    f"{'GPU ID'.ljust(id_width)}  "
    f"{'Name'.ljust(name_width)}  "
    f"{'VRAM'.ljust(vram_width)}  "
)
logger.info(header)
logger.info("-" * len(header))

for gpu in gpu_types:
    gpu_id = gpu.get("id", "Unknown")
    display_name = gpu.get("displayName", "Unknown")
    memory_gb = f"{gpu.get('memoryInGb', 'N/A')}GB"

    line = (
        f"{gpu_id.ljust(id_width)}  "
        f"{display_name.ljust(name_width)}  "
        f"{memory_gb.ljust(vram_width)}  "
    )
    logger.info(line)

# -------------------------------
# Pod selection / creation
# -------------------------------
pods = runpod.get_pods()
selected_pod = None

if pods:
    logger.info("Available pods:")
    for idx, pod in enumerate(pods):
        runtime_present = "runtime" in pod
        state = "RUNNING" if runtime_present else "STOPPED"
        logger.info(f"{idx + 1}: {pod['name']} [{state}]")

    try:
        choice = input("Select a pod number to reuse, or press Enter to skip: ").strip()
        if choice:
            idx = int(choice) - 1
            if 0 <= idx < len(pods):
                selected_pod = pods[idx]
    except Exception:
        logger.warning("Invalid selection. A new pod will be created.")

if selected_pod:
    pod_id = selected_pod["id"]
    pod_info = runpod.get_pod(pod_id)

    if not pod_info.get("runtime"):
        logger.info("Selected pod is stopped. Starting...")
        runpod.resume_pod(pod_id)
        while True:
            pod_info = runpod.get_pod(pod_id)
            if pod_info.get("runtime"):
                logger.info("Pod is now RUNNING.")
                break
            logger.info("Waiting for pod to start...")
            time.sleep(5)
else:
    confirm = input("No pod selected. Launch a new one? (y/n): ").strip().lower()
    if confirm != 'y':
        logger.info("No pod selected or created. Exiting.")
        exit(0)

    gpu_count = 1
    model_id = "tiiuae/falcon-7b-instruct"
    gpu_type_id = "NVIDIA GeForce RTX 4090"

    logger.info("Creating new pod for model: %s", model_id)

    selected_pod = runpod.create_pod(
        name="falcon-7b-instruct",
        image_name="ghcr.io/huggingface/text-generation-inference:latest",
        gpu_type_id=gpu_type_id,
        cloud_type="SECURE",
        docker_args=f"--model-id {model_id} --num-shard {gpu_count}",
        gpu_count=gpu_count,
        volume_in_gb=96,
        container_disk_in_gb=5,
        ports="80/http,29500/http",
        volume_mount_path="/data",
    )

    pod_id = selected_pod["id"]

    logger.info("Waiting for pod to become RUNNING...")
    while True:
        pod_info = runpod.get_pod(pod_id)
        if pod_info.get("runtime"):
            logger.info("Pod is now RUNNING.")
            break
        time.sleep(5)

# -------------------------------
# Inference endpoint setup
# -------------------------------
inference_url = f'https://{pod_info["id"]}-80.proxy.runpod.net'
logger.info(f"Model is live at: {inference_url}")

# -------------------------------
# FastAPI proxy server setup
# -------------------------------
app = FastAPI()

@app.post("/generate")
async def proxy(request: Request):
    try:
        data = await request.json()
        json_preview = str(data)[:100].replace("\n", " ").replace("\r", " ")
        logger.debug("Parsed request JSON preview: %s", json_preview)

        response = requests.post(f"{inference_url}/generate", json=data)
        response_preview = response.text[:100].replace("\n", " ").replace("\r", " ")
        logger.debug("Raw response preview: %s", response_preview)

        return response.json()
    except Exception as e:
        logger.error("Proxy error: %s", str(e))
        return {"error": str(e)}

def start_proxy():
    uvicorn.run(app, host="0.0.0.0", port=11435)

proxy_thread = threading.Thread(target=start_proxy, daemon=True)
proxy_thread.start()
time.sleep(2)
logger.info("Local proxy is running at http://localhost:11435/generate")

# -------------------------------
# LangChain CLI
# -------------------------------
llm = HuggingFaceTextGenInference(
    inference_server_url=inference_url,
    max_new_tokens=100,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.001,
    repetition_penalty=1.03,
)

logger.info("Enter your prompts below. Type '/bye' to exit and shut down the pod.")

try:
    while True:
        prompt = input("You: ")
        if prompt.strip().lower() == "/bye":
            logger.info("Shutting down the pod...")
            break
        try:
            output = llm(prompt)
            print("Model:", output.strip())
        except Exception as e:
            logger.error("Inference request failed: %s", str(e))
finally:
    runpod.stop_pod(pod_info["id"])
    logger.info("Pod stop requested.")
