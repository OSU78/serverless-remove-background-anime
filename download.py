# In this file, we define download_model
# It runs during container build time to get model weights built into the container
import os
import torch

import huggingface_hub
import onnxruntime as rt


def download_model():
    # do a dry run of loading the huggingface model, which will download weights at build time

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    model_path = huggingface_hub.hf_hub_download("skytnt/anime-seg", "isnetis.onnx")
    rmbg_model = rt.InferenceSession(model_path, providers=providers)
    return rmbg_model
    #scheduler = EulerDiscreteScheduler.from_pretrained(
    #   repo_id, 
    #   subfolder="scheduler", 
    #    prediction_type="epsilon"
    #)
   
    

if __name__ == "__main__":
    download_model()
