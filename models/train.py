from transformers import AutoImageProcessor, MobileNetV2ForSemanticSegmentation, Trainer, TrainingArguments
from PIL import Image
import torch
from matplotlib.colors import ListedColormap
import os
from torch.utils.data import Dataset
from torchvision import transforms
import random
import torchvision
import torch.nn.functional as F


def augmentation(image, label, angle_range=15, target_size=(512, 512)):
    #convert lable into PIL image
    label = Image.fromarray(label)

    #random horizontal flip
    if random.random() > 0.8:
        image = torchvision.transforms.functional.hflip(image)
        #print lable type
        label = torchvision.transforms.functional.hflip(label)
    
    #random vertical flip
    if random.random() > 0.5:
        image = torchvision.transforms.functional.vflip(image)
        label = torchvision.transforms.functional.vflip(label)
    
    #convert label into numpy array
    label = np.asarray(label)

    return image, label



class lpcv_dataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None, augmentation=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_filenames = sorted(os.listdir(image_folder))  # Sort filenames
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_folder, image_filename)
            
        # Generate corresponding label filename
        label_path = os.path.join(self.label_folder, image_filename)

        image = Image.open(image_path).convert("RGB")
        label = np.asarray(Image.open(label_path))[:,:,0]

        if augmentation:
            image, label = self.augmentation(image, label)

        # Scale image pixel values to [0, 1] range
        image = np.array(image) / 255.0

        # Preprocess the image using the image_processor
        inputs = image_processor(images=image, return_tensors="pt")


        #remove the 3rd dimension from input
        inputs["pixel_values"] = inputs["pixel_values"].squeeze(0)
        
        return {"pixel_values": inputs["pixel_values"], "labels": torch.tensor(label, dtype=torch.long)}

class CustomTrainer(Trainer):
    def __init__(self, *args, accuracy_tracker, **kwargs):
        super().__init__(*args, **kwargs)
        self.accuracy_tracker = accuracy_tracker

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs)
        return loss

    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        return loss

    def evaluation_step(self, model, inputs):
        loss, logits = super().evaluation_step(model, inputs)
        self.accuracy_tracker.update(inputs["labels"], logits.argmax(dim=1))
        return loss, logits

    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
      # Calculate and print accuracy metrics at the end of an epoch
        accuracy_scores = self.accuracy_tracker.get_scores()
        print(accuracy_scores)
        super().on_epoch_end(args, state, control, logs=logs, **kwargs)


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
                                                     crop_size=(512, 512))

model = MobileNetV2ForSemanticSegmentation.from_pretrained("google/deeplabv3_mobilenet_v2_1.0_513", 
                                                           num_labels=len(categories), 
                                                           ignore_mismatched_sizes=True, 
                                                           image_size=(512, 512))


transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

image_folder="LPCVC_Train_Updated/LPCVC_Train_Updated/LPCVC_Train_Updated/IMG/train"
label_folder="LPCVC_Train_Updated/LPCVC_Train_Updated/LPCVC_Train_Updated/GT_Updated/train"

val_folder="LPCVC_Val/LPCVC_Val/IMG/val"
val_label_folder="LPCVC_Val/LPCVC_Val/GT/val"

train_dataset = lpcv_dataset(image_folder, label_folder, augmentation=augmentation)

training_args = TrainingArguments(output_dir="test_trainer", num_train_epochs=10, per_device_train_batch_size=8)

eval_dataset = lpcv_dataset(val_folder, val_label_folder)
accuracy_tracker = AccuracyTracker(len(categories))

custom_trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    accuracy_tracker=accuracy_tracker
)


custom_trainer.train()