import torch
import numpy as np
from PIL import Image
from transformers import MobileViTV2ForSemanticSegmentation, AutoImageProcessor
import requests
import colorsys


def generate_colors(num_colors):
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        saturation = 0.8
        value = 0.8
        rgb_color = colorsys.hsv_to_rgb(hue, saturation, value)
        rgb_int_color = tuple(int(val * 255) for val in rgb_color)
        colors.append(rgb_int_color)
    return colors

def color_mapping(segmentation_mask, num_classes):
    colors = generate_colors(num_classes)
    
    colored_mask = np.zeros((segmentation_mask.shape[0], segmentation_mask.shape[1], 3), dtype=np.uint8)
    for class_idx in range(len(colors)):
        colored_mask[segmentation_mask == class_idx] = colors[class_idx]
    
    return Image.fromarray(colored_mask)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("apple/deeplabv3-mobilevit-small")
model = MobileViTV2ForSemanticSegmentation.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")

inputs = image_processor(images=image, return_tensors="pt")

config = model.config
#print the num class from config
print(config.num_labels)
#save config file into json
config.save_pretrained("config.json")

# Generate predictions (logits)
with torch.no_grad():
    logits = model(**inputs).logits

# Post-process logits into segmentation mask
segmentation_mask = torch.argmax(logits)

# Convert segmentation mask to colored image (assuming 3 color channels)
colored_image = color_mapping(segmentation_mask, 1000)

# Display or save the colored segmented image
colored_image.show()  # Display using a library like PIL
colored_image.save("segmented_image.png")  # Save the image
