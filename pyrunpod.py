import json
import os
import time
import logging
import threading
import requests
import runpod
import uvicorn
from fastapi import FastAPI, Request
from dotenv import load_dotenv

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
# Pod configuration
# -------------------------------
gpu_count = 1
model_id = "deepseek-ai/deepseek-coder-33b-instruct"

logger.info("Creating RunPod pod for model: %s", model_id)

# Get all my pods
pods = runpod.get_pods()

if pods:
    logger.info("Existing pods found:")
    for pod in pods:
        logger.info(f"{json.dumps(pod, indent=4)}")
else:
    logger.info("No existing pods found.")

pod = runpod.create_pod(
    name="deepseek-coder-33b-instruct",
    image_name="ghcr.io/huggingface/text-generation-inference:0.8",
    gpu_type_id="NVIDIA A100 80GB PCIe",
    cloud_type="SECURE",
    docker_args=f"--model-id {model_id} --num-shard {gpu_count}",
    gpu_count=gpu_count,
    volume_in_gb=96,
    container_disk_in_gb=5,
    ports="80/http,29500/http",
    volume_mount_path="/data",
)

# -------------------------------
# Wait for pod to be RUNNING
# -------------------------------
logger.info("Waiting for pod to reach RUNNING state...")
while True:
    pod_info = runpod.get_pod(pod["id"])
    logger.info("Pod info: %s", json.dumps(pod_info, indent=4))

    if pod_info.get("runtime"):
        logger.info("Pod is RUNNING (runtime present).")
        break

    logger.info("Pod not ready yet (runtime missing). Waiting...")
    time.sleep(5)

inference_url = f'https://{pod["id"]}-80.proxy.runpod.net/generate'
logger.info("Model is live at: %s", inference_url)

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

        response = requests.post(inference_url, json=data)

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
# Interactive CLI loop
# -------------------------------
logger.info("Enter your prompts below. Type '/bye' to exit and shut down the pod.")

try:
    while True:
        prompt = input("You: ")
        if prompt.strip().lower() == "/bye":
            logger.info("Shutting down the pod...")
            break

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.03
            }
        }

        try:
            response = requests.post("http://localhost:11435/generate", json=payload)
            result = response.json()
            output = result[0]["generated_text"] if isinstance(result, list) else result.get("generated_text", "")
            print("Model:", output.strip())
        except Exception as e:
            logger.error("Inference request failed: %s", str(e))

finally:
    logger.info("Stopping pod...")
    runpod.stop_pod(pod["id"])
    logger.info("Stopping requested!")
