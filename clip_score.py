from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from io import BytesIO

# Initialize FastAPI app
app = FastAPI()

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define request model
class MatchRequest(BaseModel):
    caption: str
    image_url: List[str]

# Helper function to load image from URL
def get_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Error loading {url}: {e}")
        return None

# Function to compute similarity score
def compute_similarity_score(image, text):
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        similarity = outputs.logits_per_image[0][0].item()

    score = (similarity / 30) * 100  # Normalize to 0–100
    return score

# Core matching function
def find_matching_image_url(image_urls, description, threshold=80):
    for url in image_urls:
        image = get_image_from_url(url)
        if image is None:
            continue
        score = compute_similarity_score(image, description)
        if score >= threshold:
            return {"url": url, "score": score}
    return None

# FastAPI endpoint
@app.post("/match-image/")
def match_image(request: MatchRequest):
    result = find_matching_image_url(request.image_url, request.caption)
    if result:
        return result
    raise HTTPException(status_code=404, detail="No matching image found")
