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
from torchvision.transforms import functional as FF
import numpy as np
import torch.nn as nn
from typing import Optional, Union, Tuple
from transformers.modeling_outputs import SemanticSegmenterOutput

class AccuracyTracker(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class**2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    def get_scores(self):
        """Returns accuracy score evaluation result.
        - overall accuracy
        - mean accuracy
        - mean IU
        - fwavacc
        """
        hist = self.confusion_matrix
        self.acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / (hist.sum(axis=1) + 0.000000001)
        self.acc_cls = np.nanmean(acc_cls)

        with np.errstate(invalid='ignore'):
            dice = 2*np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0))

        self.mean_dice = np.nanmean(dice)
        freq = hist.sum(axis=1) / hist.sum()
        self.fwavacc = (freq[freq > 0] * dice[freq > 0]).sum()
        self.cls_dice = dict(zip(range(self.n_classes), dice))

        return {
            "Overall Acc: \t": self.acc,
            "Mean Acc : \t": self.acc_cls,
            "FreqW Acc : \t": self.fwavacc,
            "Mean Dice : \t": self.mean_dice,
        }

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
    
class Mobile_segment(MobileNetV2ForSemanticSegmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(320, 14, kernel_size=1, stride=1),  # Adjust output channels to 14
            nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False)
        )


        def forward(
            self,
            pixel_values: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[tuple, SemanticSegmenterOutput]:

            output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            outputs = self.mobilenet_v2(
                pixel_values,
                output_hidden_states=True,  # we need the intermediate hidden states
                return_dict=return_dict,
            )

            encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

            logits = self.segmentation_head(encoder_hidden_states[-1])
            print(logits.shape)
            logits = self.segmentation_head(logits)

            loss = None
            if labels is not None:
                if self.config.num_labels == 1:
                    raise ValueError("The number of labels should be greater than one")
                else:
                    # upsample logits to the images' original size
                    upsampled_logits = nn.functional.interpolate(
                        logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                    )
                    loss_fct = super.CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
                    loss = loss_fct(upsampled_logits, labels)

            if not return_dict:
                if output_hidden_states:
                    output = (logits,) + outputs[1:]
                else:
                    output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return SemanticSegmenterOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states if output_hidden_states else None,
                attentions=None,
            )


class CustomTrainer(Trainer):
    def __init__(self, *args, accuracy_tracker, **kwargs):
        super().__init__(*args, **kwargs)
        self.accuracy_tracker = accuracy_tracker

    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        # Calculate and print accuracy metrics at the end of an epoch
        accuracy_scores = self.accuracy_tracker.get_scores()
        wandb.log({"Accuracy": accuracy_scores[0]})

        super().on_epoch_end(args, state, control, logs=logs, **kwargs)


    def evaluation_step(self, model, inputs):
        loss, logits = super().evaluation_step(model, inputs)
        self.accuracy_tracker.update(inputs["labels"], logits.argmax(dim=1))
        return loss, logits

def augmentation(image, label, angle_range=15, target_size=(512, 512)):
    #convert lable into PIL image
    label = Image.fromarray(label)

    #random horizontal flip
    if random.random() > 0.8:
        image = FF.hflip(image)
        #print lable type
        label = FF.hflip(label)

    #random vertical flip
    if random.random() > 0.5:
        image = FF.vflip(image)
        label = FF.vflip(label)

    #convert label into numpy array
    label = np.asarray(label)

    return image, label

def test_model(img_path, save_path, model, preprocess):

    image = Image.open(img_path)
    inputs = image_processor(images=image, return_tensors="pt")

    # Move inputs to the same device as the model's parameters
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        # Move the model to the same device as the inputs
        model = model.to(inputs["pixel_values"].device)
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

model = Mobile_segment.from_pretrained("google/deeplabv3_mobilenet_v2_1.0_513",
                                                           num_labels=len(categories),
                                                           ignore_mismatched_sizes=True,
                                                           image_size=(512, 512))


image_folder="/home/pappol/Scrivania/uni/cv/low_power_segmentation/dataset/LPCVC_Train_Updated/LPCVC_Train_Updated/LPCVC_Train_Updated/IMG/train"
label_folder="/home/pappol/Scrivania/uni/cv/low_power_segmentation/dataset/LPCVC_Train_Updated/LPCVC_Train_Updated/LPCVC_Train_Updated/GT_Updated/train"

val_folder="/home/pappol/Scrivania/uni/cv/low_power_segmentation/dataset/LPCVC_Val/LPCVC_Val/IMG/val"
val_label_folder="/home/pappol/Scrivania/uni/cv/low_power_segmentation/dataset/LPCVC_Val/LPCVC_Val/GT/val"

train_dataset = lpcv_dataset(image_folder, label_folder, augmentation=augmentation)

training_args = TrainingArguments(output_dir="test_trainer", num_train_epochs=10, per_device_train_batch_size=4)

eval_dataset = lpcv_dataset(val_folder, val_label_folder)
accuracy_tracker = AccuracyTracker(len(categories))


custom_trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    accuracy_tracker=accuracy_tracker,
)


custom_trainer.train()