import torch
import torch.quantization
from transformers import MobileNetV2ForSemanticSegmentation

# Load the saved model
model = MobileNetV2ForSemanticSegmentation.from_pretrained("google/deeplabv3_mobilenet_v2_1.0_513", 
                                                           num_labels=14, 
                                                           ignore_mismatched_sizes=True, 
                                                           image_size=(512, 512)).cuda()

# Set the model to evaluation mode
model.eval()

# Convert the model to quantized version
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save the model
torch.save(quantized_model, 'cv/low_power_segmentation/quantization/model_quantized.pt')