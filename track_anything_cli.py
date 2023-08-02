import sys 
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)) + "/")
sys.path.insert(0, os.path.join(os.path.dirname(__file__)) + "/tracker")
sys.path.insert(0, os.path.join(os.path.dirname(__file__)) + "/tracker/model")


import torch
from pathlib import Path
import cv2
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from groundingdino.util.inference import load_model, load_image, predict, annotate
from segment_anything import SamPredictor, sam_model_registry
from tracker.base_tracker import BaseTracker

import os
from os import path
from pathlib import Path

import torch
import numpy as np
from PIL import Image

model_config = {"DINO": "./preprocess/third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"}
model_weights = {"DINO" : "./preprocess/third_party/Track-Anything/checkpoints/groundingdino_swint_ogc.pth", 
                 "SAM" : "./preprocess/third_party/Track-Anything/checkpoints/sam_vit_h_4b8939.pth", 
                 "XMEM" : "./preprocess/third_party/Track-Anything/checkpoints/XMem-s012.pth"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.autograd.set_grad_enabled(False)

def extract_bbox(model, img_path, text_prompt, BOX_THRESHOLD=0.35, TEXT_THRESHOLD=0.25):
    image_source, image = load_image(img_path)
    H, W = image_source.shape[0], image_source.shape[1]
    
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

    boxes = boxes * torch.Tensor([W, H, W, H]).repeat(repeats=(boxes.shape[0], 1))

    # from xywh to xyxy
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]

    boxes = boxes.type("torch.IntTensor")
    return boxes, annotated_frame

def extract_mask(model, img_path, boxes):
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    
    track_init = np.zeros((image.shape[0], image.shape[1]))

    model.set_image(image)
    transformed_boxes = model.transform.apply_boxes_torch(boxes.to(DEVICE), image.shape[:2])
    masks, _, _ = model.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    for i, mask in enumerate(masks):
        track_init = track_init + mask[0].cpu().numpy() * (i + 1)
    
    return track_init

def extract_tracks(xmem_model, template_mask, images):
    def generator(xmem_model, template_mask, images):    
        masks = []
        logits = []
        painted_images = []
        for i in tqdm(range(len(images))):
            if i ==0:           
                mask, logit, painted_image = xmem_model.track(images[i], template_mask)
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)
                
            else:
                mask, logit, painted_image = xmem_model.track(images[i])
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)
        return masks, logits, painted_images
    
    return generator(xmem_model, template_mask, images)

def isImageFile(str):
    import re

    # Regex to check valid image file extension.
    regex = "([^\\s]+(\\.(?i)(jpe?g|png|gif|bmp))$)"

    # Compile the ReGex
    p = re.compile(regex)

    # If the string is empty
    # return false
    if str == None:
        return False

    # Return if the string
    # matched the ReGex
    if re.search(p, str):
        return True
    else:
        return False
    
parser = argparse.ArgumentParser()
parser.add_argument('--input-folder',  type=str, required=True)
parser.add_argument('--output-folder',  type=str, required=True)
parser.add_argument('--box-threshold',  type=float, default=0.35)
parser.add_argument('--text-threshold',  type=float, default=0.25)
parser.add_argument('--scale_percent',  type=float, default=0.25)
parser.add_argument('--text-prompt',  type=str, required=True)

args = parser.parse_args()

input_folder = args.input_folder 
output_folder = args.output_folder
text_prompt = args.text_prompt

scale_percent = args.scale_percent
BOX_THRESHOLD = args.box_threshold
TEXT_THRESHOLD = args.text_threshold

if not os.path.isdir(output_folder):
    os.makedirs(output_folder, exist_ok=True)

original_size = (None, None)
dim = (None, None)
images, image_paths, image_exts = [], [], []
for img in sorted(os.listdir(input_folder)):
    if not isImageFile(img):
        continue 
    
    img_path = input_folder + "/" + "/" + img
    img_ext = str(Path(img).with_suffix(""))
    
    img = cv2.imread(img_path)
    
    original_size = (img.shape[1], img.shape[0])
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)

    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                
    images.append(img)
    image_paths.append(img_path)
    image_exts.append(img_ext)
    

dino_model = load_model(model_config["DINO"], model_weights["DINO"]).to(DEVICE)
sam_model = SamPredictor(sam_model_registry["vit_h"](checkpoint=model_weights["SAM"]).to(DEVICE))
xmem_model = BaseTracker(model_weights["XMEM"], device=DEVICE)

boxes, annotated_frame = extract_bbox(dino_model, image_paths[0], text_prompt, BOX_THRESHOLD, TEXT_THRESHOLD)    
track_init = extract_mask(sam_model, image_paths[0], boxes)

track_init = cv2.resize(track_init, dim, interpolation=cv2.INTER_AREA)
masks, logits, painted_images = extract_tracks(xmem_model, track_init, images)

for mask, painted_img, img_ext in zip(masks, painted_images, image_exts):
    mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_AREA)
    np.save("{}/{}.npy".format(output_folder, img_ext), mask)

    painted_img = cv2.resize(painted_img, original_size, interpolation=cv2.INTER_AREA)
    cv2.imwrite("{}/{}.png".format(output_folder, img_ext), painted_img)

