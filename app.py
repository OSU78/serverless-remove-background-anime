import os
import torch
import base64
from io import BytesIO
from torch import autocast
import random
import onnxruntime as rt
import numpy as np
import cv2
import huggingface_hub


# providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
# model_path = huggingface_hub.hf_hub_download("skytnt/anime-seg", "isnetis.onnx")
# rmbg_model = rt.InferenceSession(model_path, providers=providers)
# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model

def base64_to_numpy(string):
    img_data = base64.b64decode(string)
    print(string)
    img_arr = np.frombuffer(img_data, np.uint8)
    return img_arr


def get_mask(img, s=1024):
    img = (img / 255).astype(np.float32)
    print(img.shape)
    h, w = h0, w0 = img.shape[:-1]
    h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
    ph, pw = s - h, s - w
    img_input = np.zeros([s, s, 3], dtype=np.float32)
    img_input[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(img, (w, h))
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = img_input[np.newaxis, :]
    mask = model.run(None, {'img': img_input})[0][0]
    mask = np.transpose(mask, (1, 2, 0))
    mask = mask[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
    mask = cv2.resize(mask, (w0, h0))[:, :, np.newaxis]
    return mask



def rmbg_fn(img):
    #print (img)
    mask = get_mask(img)
    img = (mask * img + 255 * (1 - mask)).astype(np.uint8)
    mask = (mask * 255).astype(np.uint8)
    img = np.concatenate([img, mask], axis=2, dtype=np.uint8)
    mask = mask.repeat(3, axis=2)
    #print(img)
    return mask, img


def opcv2_to_base64(img):
   img = cv2.imread('test.jpg')
   _, im_arr = cv2.imencode('.jpg', img)  # im_arr: image in Numpy one-dim array format.
   im_bytes = im_arr.tobytes()
   img_b64 = base64.b64encode(im_bytes)
   return img_b64

def encode_image(img):
    img_bytes = cv2.imencode('.jpg', img)[1].tostring()
    img_b64 = base64.b64encode(img_bytes)
    img_str = img_b64.decode('utf-8')
    return img_str


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    image = model_inputs
   
    im_bytes = base64.b64decode(image)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    #print(img_arr)
   
    if image != None:
        buffered = BytesIO()
        # image.save(buffered,format="png")
        # image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        output_mask , output_result  = rmbg_fn(img)
        # Return the results as a dictionary
        return {'output_mask':encode_image(output_mask),'output_result' :encode_image(output_result)}



