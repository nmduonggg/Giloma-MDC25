import math
import uuid

import numpy as np
import PIL.Image
import PIL.ImageDraw
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size=256):
    return A.Compose([
        # Ensure the target remains centered by limiting spatial transformations
        A.Resize(height=image_size, width=image_size),
        # A.LongestMaxSize(max_size=224),
        
        # A.PadIfNeeded(224, 224),
        
        # Horizontal and Vertical flips
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        
        # Random rotations around the center
        A.Rotate(limit=15, p=0.5, border_mode=0),
        
        # Elastic transformations can help with subtle deformations
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        
        # Random brightness and contrast adjustments
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        
        # Adding noise
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        
        # Normalize the image (assuming images are in [0, 255])
        A.Normalize(mean=[0.485, 0.456, 0.406],    # Normalize the tensor with mean and std
                         std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        
        # Convert to tensor
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

def get_valid_transforms(image_size=256):
    return A.Compose([
        # Resize to ensure consistency
        A.Resize(height=image_size, width=image_size),
        # A.LongestMaxSize(max_size=224),
        
        # A.PadIfNeeded(224, 224),
        # Normalize the image
        A.Normalize(mean=[0.485, 0.456, 0.406],    # Normalize the tensor with mean and std
                         std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        
        # Convert to tensor
        ToTensorV2(),
    ])

CELL_TRANSFORMS = transforms.Compose([
    transforms.Resize((240, 240)),                     # Resize the image to 256x256 pixels
    transforms.RandomCrop(224),                         # Randomly crop a 224x224 region
    transforms.ToTensor(),                              # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],    # Normalize the tensor with mean and std
                         std=[0.229, 0.224, 0.225])
])

# CELL_TRANSFORMS = transforms.Compose([
#     transforms.Resize((240, 240)),                     # Resize the image to 256x256 pixels
#     transforms.RandomCrop(224),                         # Randomly crop a 224x224 region
#     transforms.ToTensor(),                              # Convert the image to a PyTorch tensor
#     transforms.Normalize(mean=[0.5059, 0.3313, 0.5922],    # Normalize the tensor with mean and std
#                          std=[0.2063, 0.1800, 0.1496])
# ])

ROI_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),                     # Resize the image to 256x256 pixels
    transforms.ToTensor(),                              # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],    # Normalize the tensor with mean and std
                         std=[0.229, 0.224, 0.225])
])

TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),                      # Resize the image to 256x256 pixels                     # Crop the center 224x224 region
    transforms.ToTensor(),                               # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],     # Normalize the tensor with mean and std
                         std=[0.229, 0.224, 0.225])
])

# TEST_TRANSFORMS = transforms.Compose([
#     transforms.Resize((224, 224)),                      # Resize the image to 256x256 pixels                     # Crop the center 224x224 region
#     transforms.ToTensor(),                               # Convert the image to a PyTorch tensor
#     transforms.Normalize(mean=[0.5059, 0.3313, 0.5922],    # Normalize the tensor with mean and std
#                          std=[0.2063, 0.1800, 0.1496])
# ])

## normal function
def horizontal_flip(img):
    flipper = transforms.RandomHorizontalFlip(p=1)
    return flipper(img)

def shape_to_mask(img_shape, points, shape_type=None, line_width=10, point_size=5):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask
    
def get_bounding_box(polygon_points, height, width, offset=10):
    """
    Calculates the bounding box of a polygon.

    :param polygon_points: List of tuples representing the polygon points [(x1, y1), (x2, y2), ...]
    :return: A tuple representing the bounding box (min_x, min_y, max_x, max_y)
    """
    # Extract x and y coordinates separately
    x_coordinates = [point[0] for point in polygon_points]
    y_coordinates = [point[1] for point in polygon_points]

    # Find min and max for x and y
    min_x = max(min(x_coordinates) - offset, 0)
    max_x = min(max(x_coordinates) + offset, width)
    min_y = max(min(y_coordinates) - offset, 0)
    max_y = min(max(y_coordinates) + offset, height)

    return min_x, min_y, max_x, max_y

def single_shape_to_label(img_shape, points, label_name_to_value):
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []
    
    label = "blank" # placeholder
    
    group_id = uuid.uuid1()
    shape_type = "polygon"

    cls_name = label
    instance = (cls_name, group_id)

    if instance not in instances:
        instances.append(instance)
    ins_id = instances.index(instance) + 1
    cls_id = label_name_to_value[cls_name]

    mask = shape_to_mask(img_shape[:2], points, shape_type)
    cls[mask] = cls_id
    ins[mask] = ins_id
        
    return cls, ins

def shapes_to_label(img_shape, shapes, label_name_to_value):
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []
    for shape in shapes:
        points = shape["points"]
        label = shape["label"]
        group_id = shape.get("group_id")
        if group_id is None:
            group_id = uuid.uuid1()
        shape_type = shape.get("shape_type", None)

        cls_name = label
        instance = (cls_name, group_id)

        if instance not in instances:
            instances.append(instance)
        ins_id = instances.index(instance) + 1
        cls_id = label_name_to_value[cls_name]

        mask = shape_to_mask(img_shape[:2], points, shape_type)
        cls[mask] = cls_id
        ins[mask] = ins_id
        
        
    return cls, ins

def shapes_to_independent_labels(img, shapes, offset=10, global_offset=50):
    
    height, width = img.shape[:2]
    crops = []
    for shape in shapes:
        points = shape["points"]
        label = shape["label"]
        shape_type = shape.get("shape_type", None)
        
        if isinstance(img, np.ndarray):
            img = PIL.Image.fromarray(img)
        
        bbox = get_bounding_box(points, height, width, offset)
        global_bbox = get_bounding_box(points, height, width, global_offset)
        
        cropped = img.crop(bbox)
        global_cropped = img.crop(global_bbox)
        crops.append((cropped, global_cropped, label, points))
        
    return crops

        
        
