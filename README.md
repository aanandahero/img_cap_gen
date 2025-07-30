# 🧠 Image Caption Generator

This project uses a pre-trained BLIP (Bootstrapped Language Image Pretraining) model to generate natural-language captions for input images.

## 🔧 Technologies
- Python
- HuggingFace Transformers
- PyTorch
- PIL (Pillow)
- BLIP Model (`Salesforce/blip-image-captioning-base`)

## 🚀 How to Run
1. Install dependencies:
pip install transformers torch torchvision pillow requests

2. Run the script and replace the image URL or path with your own image.

## 📷 Example
**Input Image:** A cat lying on a couch  
**Output Caption:** `"a cat laying on a couch"`

## 💡 Use Cases
- Assistive technology for the visually impaired
- Social media auto-captioning
- E-commerce image indexing

## 📌 Note
This is an inference-only project using a pre-trained model. No training is required.
