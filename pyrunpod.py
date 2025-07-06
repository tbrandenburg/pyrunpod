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
    level=logging.DEBUG,  # Set to INFO or DEBUG as needed
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
gpu_count = 2
model_id = "deepseekcoder/33b"

logger.info("Creating RunPod pod for model: %s", model_id)
pod = runpod.create_pod(
    name="deepseekcoder-33b",
    image_name="ghcr.io/huggingface/text-generation-inference:0.8",
    gpu_type_id="NVIDIA A100 80GB PCIe",
    cloud_type="SECURE",
    docker_args=f"--model-id {model_id} --num-shard {gpu_count}",
    gpu_count=gpu_count,
    volume_in_gb=195,
    container_disk_in_gb=5,
    ports="80/http,29500/http",
    volume_mount_path="/data",
)

# -------------------------------
# Wait for pod to be RUNNING
# -------------------------------
logger.info("Waiting for pod to reach RUNNING state...")
while True:
    status = runpod.get_pod(pod["id"])["status"]
    if status == "RUNNING":
        break
    logger.info("Pod status: %s", status)
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
            response = requests.post("http://localhost:8000/generate", json=payload)
            result = response.json()
            output = result[0]["generated_text"] if isinstance(result, list) else result.get("generated_text", "")
            print("Model:", output.strip())
        except Exception as e:
            logger.error("Inference request failed: %s", str(e))

finally:
    logger.info("Deleting pod...")
    runpod.delete_pod(pod["id"])

    for _ in range(30):  # max wait ~60 seconds
        try:
            current_status = runpod.get_pod(pod["id"])
            logger.info("Waiting for deletion... current status: %s", current_status['status'])
            time.sleep(2)
        except Exception:
            logger.info("Pod deleted successfully.")
            break
    else:
        logger.warning("Timeout while waiting for pod deletion. Check RunPod dashboard.")

    logger.info("Shutdown complete.")
