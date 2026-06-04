from transformers import pipeline
from PIL import Image

# Load pretrained model
detector = pipeline(
    "image-classification",
    model="dima806/deepfake_vs_real_image_detection"
)

# Load image
image = Image.open("test.jpg")

# Predict
result = detector(image)

print(result)