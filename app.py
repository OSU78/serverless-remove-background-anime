import os
import torch
import base64
from io import BytesIO
from torch import autocast
import random
import onnxruntime as rt
import numpy as np
import cv2
# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global rmbg_model




def get_mask(img, s=1024):
    img = (img / 255).astype(np.float32)
    h, w = h0, w0 = img.shape[:-1]
    h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
    ph, pw = s - h, s - w
    img_input = np.zeros([s, s, 3], dtype=np.float32)
    img_input[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(img, (w, h))
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = img_input[np.newaxis, :]
    mask = rmbg_model.run(None, {'img': img_input})[0][0]
    mask = np.transpose(mask, (1, 2, 0))
    mask = mask[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
    mask = cv2.resize(mask, (w0, h0))[:, :, np.newaxis]
    return mask


def rmbg_fn(img):
    mask = get_mask(img)
    img = (mask * img + 255 * (1 - mask)).astype(np.uint8)
    mask = (mask * 255).astype(np.uint8)
    img = np.concatenate([img, mask], axis=2, dtype=np.uint8)
    mask = mask.repeat(3, axis=2)
    return mask, img


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    image = model_inputs.get('image', None)
   
    
    if image != None:
        buffered = BytesIO()
        # image.save(buffered,format="png")
        # image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        output_mask , output_result  = rmbg_fn(image)
        # Return the results as a dictionary
        return {'output_mask':output_mask,'output_result' : output_result}



