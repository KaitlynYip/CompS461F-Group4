import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import timm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os

# --- Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "general_model_final.pth"

# --- 1. Model Definitions ---
# Text Encoder
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_model = BertModel.from_pretrained('bert-base-uncased').to(device)
text_model.eval()

# Vision Model (DINOv2)
image_model = timm.create_model(
    'vit_small_patch14_dinov2.lvd142m', 
    pretrained=True, 
    num_classes=0,
    img_size=224
).to(device)
image_model.eval()

# Projection Layers
projection = nn.Linear(384, 768).to(device)
logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

# --- 2. Load Your Weights ---
try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # Use the EXACT keys from your original save version
    image_model.load_state_dict(checkpoint['image_model_state_dict'])
    projection.load_state_dict(checkpoint['projection_state_dict'])
    
    if 'logit_scale' in checkpoint:
        logit_scale.data = checkpoint['logit_scale']
        
    print("✅ Weights loaded using original long-form keys!")
except Exception as e:
    print(f"⚠️ Error: {e}")

# --- 3. Inference Logic ---
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def encode_texts(prompts):
    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = text_model(**inputs)
    return outputs.last_hidden_state[:, 0, :]

def predict(input_img, labels_text):
    # CHANGED: Return an empty dict if inputs are missing 
    # to avoid the "float_parsing" error in gr.Label
    if input_img is None or not labels_text or labels_text.strip() == "":
        return {} 
    
    try:
        # Prepare Text
        classes = [c.strip() for c in labels_text.split(",")]
        # Extra safety: filter out empty strings from the list
        classes = [c for c in classes if c]
        
        prompts = [f"a photo of a {c}" for c in classes]
        text_features = F.normalize(encode_texts(prompts), dim=-1)
        
        # Prepare Image
        img_t = transform(input_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            feats = image_model(img_t)
            if feats.ndim == 3:
                feats = feats[:, 0, :]
                
            image_features = F.normalize(projection(feats), dim=-1)
            
            # Calculate Similarity
            logits = logit_scale.exp() * (image_features @ text_features.T)
            probs = F.softmax(logits, dim=-1).cpu().numpy().squeeze()
            
        # Handle case with only one label (ensure it's an array for indexing)
        if len(classes) == 1:
            return {classes[0]: float(probs)}
            
        return {classes[i]: float(probs[i]) for i in range(len(classes))}
    
    except Exception as e:
        # Log the error to the console, but return empty dict to UI 
        # so it doesn't show the "System Error" crash
        print(f"Prediction Error: {e}")
        return {}

# --- 4. Gradio UI ---
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="Labels (comma separated)", placeholder="e.g. cat, shark, ocean")
    ],
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title="🚀 Zero-Shot Classification (DINOv2 + BERT)",
    description="Final Year Project Demo: Improved Zero-Shot Classification through Vision-Language Fusion.",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()