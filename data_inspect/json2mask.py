import numpy as np
import json
import base64
import cv2
from PIL import Image

import os
import os.path as osp
import PIL
from manual_prompt import *

import utils
from labelme import utils as lbl_utils
from tqdm import tqdm
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor


import base64
# from openai import OpenAI

# client = OpenAI()
# Load the model in half-precision on the available device(s)
# llm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", device_map="auto")
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# def generate_visual_prop_qwen(image_path, label):

#     # Image
#     image = Image.open(image_path)
#     image = image.resize((336, 336))

#     conversation = [
#         {
#             "role":"user",
#             "content":[
#                 {
#                     "type":"image",
#                 },
#                 {
#                     "type":"text",
#                     "text": DESCRIBE_PROMPT
#                 }
#             ]
#         }
#     ]


#     # Preprocess the inputs
#     text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
#     # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

#     inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
#     inputs = inputs.to('cuda')

#     # Inference: Generation of the output
#     output_ids = llm_model.generate(**inputs, max_new_tokens=2000)
#     generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
#     content = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
#     # feature_ans_pairs = {}
#     # print(content)
#     # feature_ans_pairs = {
#     #     k.strip().lower(): {'text': v.strip().lower().replace(' ', '_'), 'binary': int(idx.strip())}
#     #     for pair in content.split('\n') if ':' in pair
#     #     for k, v, idx in [pair.split(':', 2)]  # Ensures only three values are extracted
#     # }
#     # feature_ans_pairs = {k.replace(' ', '_'): v for k, v in feature_ans_pairs.items()}
#     # return feature_ans_pairs
#     return content

# def generate_visual_prop(image_path):
#     # Getting the Base64 string
#     base64_image = encode_image(image_path)
    
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {
#                 "role": "developer",
#                 "content": SYSTEM_PROMPT
#             },
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": PROP_PROMPT,
#                     },
#                     {
#                         "type": "image_url",
#                         "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
#                     },
#                 ],
#             }
#         ],
#     )

#     content = response.choices[0].message.content
#     print(content)
#     # feature_ans_pairs = {k: v for k, v in pair.split(':') for pair in content.split('/n')}
#     feature_ans_pairs = {}
#     try:
#         feature_ans_pairs = {
#             k.strip().lower(): {'text': v.strip().lower().replace(' ', '_'), 'binary': int(idx.strip())}
#             for pair in content.split('\n') if ':' in pair
#             for k, v, idx in [pair.split(':', 2)]  # Ensures only three values are extracted
#         }
#         feature_ans_pairs = {k.replace(' ', '_'): v for k, v in feature_ans_pairs.items()}
#     except:
#         print(content)
#     return feature_ans_pairs
    
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

def convert2mask(json_file, out_dir, prefix, gen_visual=False):
    global gid
    
    data = json.load(open(json_file))
    imageData = data.get("imageData")

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
            label_name_to_value[label_name] = label_value
    lbl, _ = utils.shapes_to_label(img.shape, data["shapes"], label_name_to_value)
    
    independent_crops = utils.shapes_to_independent_labels(img, data['shapes'], old_cut=False)    # (PIL image, label)

    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name
        
    # lbl_viz = imgviz.label2rgb(
    #     lbl, imgviz.asgray(img)
    # )

    # PIL.Image.fromarray(img).save(osp.join('./', f"img_{prefix}.png"))
    # lbl_utils.lblsave(osp.join('./', f"label_{prefix}.png"), lbl)
    # PIL.Image.fromarray(lbl_viz).save(osp.join('./', f"labelviz_{prefix}.png"))
    
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
        # if gen_visual: 
        #     feature_ans = generate_visual_prop_qwen(roi_path, label)
        #     data['visual_property'] = feature_ans
        gid += 1
        
    return datas

if __name__=='__main__':
    
    gid = 0
    raw_dir = '/home/nmduongg/Gilioma-ISBI25/DATA/Glioma_MDC_2025_test'
    out_dir = '/home/nmduongg/Gilioma-ISBI25/PROCESSED_DATA/OneShotTesting/Reprocess_PublicTesting_to_Check'
    prefix = 'real_testing'
    
    out_prefix_dir = os.path.join(out_dir, prefix)
    os.makedirs(out_prefix_dir, exist_ok=True)
    
    he_cnt = 0
    datas = []
    
    fns = list()
    for fn in os.listdir(raw_dir):
        if '.json' in fn: fns.append(fn)
    num_instances = len(fns)
    
    # for fn in os.listdir(raw_dir):
    for i in tqdm(range(num_instances), total=num_instances):
        # fn = f"{prefix}{(i+1):04d}.json"
        ## For One Shot testing
        fn = f"testing{(i+1):04d}.json"
        # if not fn.endswith('.json'): continue
        path = os.path.join(raw_dir, fn)
    
        data = convert2mask(path, out_dir=out_prefix_dir, prefix=prefix, gen_visual=False)
        datas += data
        he_cnt += 1
    
        if i % 10 == 0:
            json_save_fn = os.path.join(out_dir, f"{prefix if prefix == 'real_testing' else 'all'}_data.json")
            with open(json_save_fn, 'w') as f:
                json.dump(datas, f, indent=4)
                
    json_save_fn = os.path.join(out_dir, f"{prefix if prefix == 'real_testing' else 'all'}_data.json")
    with open(json_save_fn, 'w') as f:
        json.dump(datas, f, indent=4)
        
    
        
    print(f"[FINAL] Convert {he_cnt} images into {len(datas)} patches")