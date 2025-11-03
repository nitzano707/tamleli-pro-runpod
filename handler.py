import runpod
from app import process_request

def handler(event):
    return process_request(event)

runpod.serverless.start({"handler": handler})
