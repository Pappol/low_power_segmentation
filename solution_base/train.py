from transformers import AutoImageProcessor, MobileNetV2ForSemanticSegmentation, Trainer, TrainingArguments
from PIL import Image
import torch
from matplotlib.colors import ListedColormap
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision import transforms
import random

def augmentation(image, label, angle_range=15, target_size=(512, 512)):
    # Random rotation
    if random.random() > 0.8:
        angle = random.uniform(-angle_range, angle_range)
        image = transforms.functional.rotate(image, angle, resample=Image.BILINEAR)
        label = transforms.functional.rotate(label, angle, resample=Image.BILINEAR)
        #fill the space with black color
        image = transforms.functional.resize(image, target_size, Image.BILINEAR)
        #fill the lable with background color
        label = transforms.functional.resize(label, target_size, Image.NEAREST) #todo check if it is correct

    #random horizontal flip
    if random.random() > 0.8:
        image = transforms.functional.hflip(image)
        label = transforms.functional.hflip(label)
    
    #random vertical flip
    if random.random() > 0.5:
        image = transforms.functional.vflip(image)
        label = transforms.functional.vflip(label)
    

    return image, label



class lpcv_dataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_filenames = sorted(os.listdir(image_folder))  # Sort filenames
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_folder, image_filename)
        
        # Generate corresponding label filename
        label_path = os.path.join(self.label_folder, image_filename)

        image = Image.open(image_path).convert("RGB")
        label = np.asarray(Image.open(label_path))[:,:,0]

        if self.transform:
            image = self.transform(image)

        return {"image": image, "label": label}
    

def test_model(img_path, save_path, model, preprocess):
    image = Image.open(img_path)
    inputs = image_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    # Post-process logits into segmentation mask
    segmentation_mask = torch.argmax(logits, dim=1)

    # Convert segmentation mask to colored image (assuming 14 color channels)
    colored_image = ListedColormap(colors)(segmentation_mask[0].cpu().numpy())

    #save the image
    Image.fromarray((colored_image * 255).astype(np.uint8)).save("segmented_image.png")





categories = ["background", "avalanche",
              "building_undamaged", "building_damaged",
              "cracks/fissure/subsidence", "debris/mud/rock flow",
              "fire/flare", "flood/water/river/sea",
              "ice_jam_flow", "lava_flow",
              "person", "pyroclastic_flow",
              "road/railway/bridge", "vehicle"]

colors = ['black', 'white', 'pink', 'yellow', 'orange', 'brown',
          'red', 'blue', 'navy', 'orange', 'cyan', 'gray',
          'magenta']

#import changing initial resulution and number of classes
image_processor = AutoImageProcessor.from_pretrained("google/deeplabv3_mobilenet_v2_1.0_513", 
                                                     num_labels=len(categories), 
                                                     ignore_mismatched_sizes=True, 
                                                     image_size=(512, 512))

model = MobileNetV2ForSemanticSegmentation.from_pretrained("google/deeplabv3_mobilenet_v2_1.0_513", 
                                                           num_labels=len(categories), 
                                                           ignore_mismatched_sizes=True, 
                                                           image_size=(512, 512))

transforms = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    transforms.ToTensor(),
])


