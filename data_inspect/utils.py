import math
import uuid

import numpy as np
import PIL.Image
import PIL.ImageDraw

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
    
def get_bounding_box(polygon_points, height, width):
    """
    Calculates the bounding box of a polygon.

    :param polygon_points: List of tuples representing the polygon points [(x1, y1), (x2, y2), ...]
    :return: A tuple representing the bounding box (min_x, min_y, max_x, max_y)
    """
    # Extract x and y coordinates separately
    x_coordinates = [point[0] for point in polygon_points]
    y_coordinates = [point[1] for point in polygon_points]

    # Find min and max for x and y
    min_x = max(min(x_coordinates) - 20, 0)
    max_x = min(max(x_coordinates) + 20, width)
    min_y = max(min(y_coordinates) - 20, 0)
    max_y = min(max(y_coordinates) + 20, height)

    return min_x, min_y, max_x, max_y

# # Example usage
# polygon_points = [(2, 3), (5, 11), (9, 5), (12, 8), (5, 6)]
# bounding_box = get_bounding_box(polygon_points, 100, 100)
# print("Bounding Box:", bounding_box)


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

def shapes_to_independent_labels(img, shapes, label_name_to_value):
    
    height, width = img.shape[:2]
    crops = []
    for shape in shapes:
        points = shape["points"]
        label = shape["label"]
        shape_type = shape.get("shape_type", None)
        
        if isinstance(img, np.ndarray):
            img = PIL.Image.fromarray(img)
        
        bbox = get_bounding_box(points, height, width)
        
        cropped = img.crop(bbox)
        crops.append((cropped, label, points))
        
    return crops
        
        