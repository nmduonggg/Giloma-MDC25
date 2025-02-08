import os
import numpy as np
import copy
import json
import torch, cv2, celldetection as cd

import labelme.utils as lbl_utils
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

np.random.seed(123)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model():
    # Load pretrained model
    model = cd.fetch_model('ginoro_CpnResNeXt101UNet-fbe875f1a3e5ce2c', check_hash=True).to(device)
    model.score_thresh=0.8
    model.nms_thresh=0.2
    model.eval()
    
    return model

def load_image_data(data_json):
    imageData = data_json.get("imageData")
    img = lbl_utils.img_b64_to_arr(imageData)
    return img

def infer_single_image(img, model):
    # Load input
    img = Image.fromarray(img)
    img = np.array(img.convert('RGB'))

    # Run model
    with torch.no_grad():
        x = cd.to_tensor(img, transpose=True, device=device, dtype=torch.float32)
        x = x / 255  # ensure 0..1 range
        x = x[None]  # add batch dimension: Tensor[3, h, w] -> Tensor[1, 3, h, w]
        y = model(x)

    # Show results for each batch item
    contours = y['contours'][0]
    # contours = [cv2.approxPolyDP(contour.cpu().numpy().astype('int'), epsilon=0.01, closed=True) for contour in contours]
    
    return contours

def fill_contours_to_template(data_json: dict, contours: list):
    # template is obtained from json
    template = copy.deepcopy(data_json)
    shapes = list()
    for idx, contour in enumerate(contours):
        contour = contour.squeeze(1)
        shape_data = {
            'label': f"Blank{idx+1}",
            'points': contour.tolist(),
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
            "mask": None
        }
        
        shapes.append(shape_data)
        
    template['shapes'] = shapes
    return template


def infer_all(folder, model, out_folder='./'):
    json_fns = list()
    for fn in os.listdir(folder):
        if '.json' in fn: json_fns.append(fn)
        
    print(f"Processing: {len(json_fns)} json file for semi-supervise phase.")
        
    json_fns.sort()
        
    for json_fn in tqdm(json_fns, total=len(json_fns)):
        json_path = os.path.join(folder, json_fn)
        data_json = json.load(open(json_path))
        
        image = load_image_data(data_json)
        contours = infer_single_image(image, model)
        new_data_json = fill_contours_to_template(data_json, contours)
        
        saved_json_path = os.path.join(out_folder, json_fn.replace("training", "real_testing"))
        with open(saved_json_path, 'w') as f:
            json.dump(new_data_json, f, indent=4)
        
        
def main():
    folder = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/_RAW_DATA/122824/Data_122824/Glioma_MDC_2025_training'
    out_folder = '/home/manhduong/ISBI25_Challenge/Giloma-MDC25/_PROCESSED_DATA/semi_supervise/raw_format'
    os.makedirs(out_folder, exist_ok=True)
    
    model = load_model()
    infer_all(folder, model, out_folder)
    
if __name__=='__main__':
    main()
        