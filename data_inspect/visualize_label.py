import numpy as np
import json
import base64
import cv2
import imgviz

import os
import os.path as osp
import PIL

import utils
from labelme import utils as lbl_utils

def blend_images(image1, image2, alpha, beta, gamma=0.0):
    """
    Blends two images together using weighted addition.

    Parameters:
        image1_path (str): Path to the first image.
        image2_path (str): Path to the second image.
        alpha (float): Weight for the first image. (0.0 to 1.0)
        beta (float): Weight for the second image. (0.0 to 1.0)
        gamma (float, optional): Scalar added to the weighted sum. Defaults to 0.0.

    Returns:
        blended_image (numpy.ndarray): The blended image.
    """

    if image1 is None or image2 is None:
        raise ValueError("One or both images could not be loaded. Check the file paths.")

    # Ensure both images have the same dimensions
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions to blend.")

    # Blend the images
    blended_image = cv2.addWeighted(image1, alpha, image2, beta, gamma)

    return blended_image

def convert2mask(json_file, out_dir, prefix):
    global gid
    
    data = json.load(open(json_file))
    imageData = data.get("imageData")
    
    height, width = data['imageHeight'], data['imageWidth']

    if not imageData:
        imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
        with open(imagePath, "rb") as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode("utf-8")
    img = lbl_utils.img_b64_to_arr(imageData)

    label_name_to_value = {"_background_": 0}
    for shape in sorted(data["shapes"], key=lambda x: x["label"]):
        label_name = shape["label"]
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = 1
    lbl, _ = utils.shapes_to_label(img.shape, data["shapes"], label_name_to_value)
    
    independent_crops = utils.shapes_to_independent_labels(img, data['shapes'], old_cut=False)    # (PIL image, label)

    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = "Blank"
        
    color_map = np.array([
        [255, 255, 255],
        [0, 255, 0]
    ], dtype=np.uint8)
    lbl_viz = imgviz.label2rgb(
        lbl, img, label_names=None, alpha=0.2,
        colormap=color_map
    )

    # os.makedirs('./examples', exist_ok=True)
    # PIL.Image.fromarray(img).save(osp.join('./examples', f"img_{gid}.png"))
    # lbl_utils.lblsave(osp.join('./examples', f"label_semi_{gid}.png"), lbl)
    # PIL.Image.fromarray(lbl_viz).save(osp.join('./examples', f"labelviz_semi_{gid}.png"))
    
    datas = []
    for image, label, points in independent_crops:
        data = {
            'id': gid,
            'original_json_file': json_file,
            'label': label,
            'points': points
        }
        datas.append(data)
        roi_path = os.path.join(out_dir, f"{prefix}_{gid}.jpg")
        image.save(roi_path)
        gid += 1
    
    return datas

if __name__=='__main__':
    
    gid = 0
    # raw_dir = '/home/nmduongg/Gilioma-ISBI25/DATA/Glioma_MDC_2025_training'
    raw_dir = '/home/nmduongg/Gilioma-ISBI25/PROCESSED_DATA/semi_supervise/raw_format'
    out_dir = '.'
    # prefix = 'training'
    prefix = 'real_testing'
    
    out_prefix_dir = os.path.join(out_dir, prefix)
    os.makedirs(out_prefix_dir, exist_ok=True)
    
    he_cnt = 0
    datas = []
    
    fns = [fn for fn in os.listdir(raw_dir)]
    num_instances = len(fns) // 2
    
    # for fn in os.listdir(raw_dir):
    for i in range(num_instances):
        if i not in [13]: continue
        fn = f"{prefix if 'real' not in prefix else 'real_testing'}{(i+1):04d}.json"
        # if not fn.endswith('.json'): continue
        path = os.path.join(raw_dir, fn)
    
        data = convert2mask(path, out_dir=out_prefix_dir, prefix=prefix)
        datas += data
        he_cnt += 1
        gid += 1
        
        if gid == 20: break
    
    # json_save_fn = os.path.join(out_dir, f"{prefix if prefix == 'real_testing' else 'all'}_data.json")
    # with open(json_save_fn, 'w') as f:
    #     json.dump(datas, f, indent=4)
    
        
    # print(f"[FINAL] Convert {he_cnt} images into {len(datas)} patches")
    
    # with open(path, 'r') as f:
    #     data = json.load(f)
        
    # original_image = cv2.imread(image_path)

    # # Extract points from the JSON
    # shapes = data['shapes']
    # mask_height, mask_width = data["imageHeight"], data["imageWidth"]
    # # mask_height, mask_width = 512, 512
    # mask = np.zeros((mask_height, mask_width), dtype=np.uint8)

    # for shape in shapes:
    #     name = shape['label']
    #     print(name)
    #     points = np.array(shape['points'], dtype=np.int32)

    #     # Define the dimensions of the mask (example: 512x512)

    #     # Draw the polygon on the mask
    #     color = 255 if name=='Mitosis' else 50
    #     cv2.fillPoly(mask, [points], color=255)  # Use 255 for white
        
    # original_image = cv2.resize(original_image, (mask_width, mask_height))
    # height, width = original_image.shape[:2]
    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # mask = cv2.resize(mask, (width, height))

    # # Save the mask image (optional)
    # cv2.imwrite("polygon_mask.jpg", mask)

    # print(mask.shape, original_image.shape)
    # blended = blend_images(original_image, mask, alpha=0.5, beta=0.5)
    # cv2.imwrite("blended_image.jpg", blended)