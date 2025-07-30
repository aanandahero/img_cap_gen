from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import torch

# Load model & processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load an image from URL or file
image_url = "https://images.pexels.com/photos/104827/cat-pet-animal-domestic-104827.jpeg"
image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')

# Process image
inputs = processor(images=image, return_tensors="pt")
out = model.generate(**inputs)

# Generate caption
caption = processor.decode(out[0], skip_special_tokens=True)
print("Caption:", caption)
