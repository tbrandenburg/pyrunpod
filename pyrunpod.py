import json
import os
import time
import logging
import runpod
from dotenv import load_dotenv
from langchain.llms import HuggingFaceTextGenInference

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

# -------------------------------
# Setup LangChain LLM
# -------------------------------
inference_server_url = f'https://{pod["id"]}-80.proxy.runpod.net'
llm = HuggingFaceTextGenInference(
    inference_server_url=inference_server_url,
    max_new_tokens=100,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.001,
    repetition_penalty=1.03,
)

logger.info("Model is live at: %s", inference_server_url)
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
    logger.info("Stopping pod...")
    runpod.stop_pod(pod["id"])
    logger.info("Stopping requested!")
