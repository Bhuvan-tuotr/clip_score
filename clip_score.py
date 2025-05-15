from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from io import BytesIO

# Load the CLIP model and processor once
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize FastAPI
app = FastAPI(title="CLIP Image Matcher")

# Request model
class MatchRequest(BaseModel):
    image_urls: List[str]
    description: str
    threshold: Optional[float] = 80.0

# Helper function to fetch and process an image
def get_image_from_url(url: str) -> Optional[Image.Image]:
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Error loading {url}: {e}")
        return None

# Compute similarity score between image and text
def compute_similarity_score(image: Image.Image, text: str) -> float:
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        similarity = outputs.logits_per_image[0][0].item()
    return (similarity / 30) * 100  # Normalize to 0–100

# FastAPI endpoint
@app.post("/match-image")
def find_matching_image(req: MatchRequest):
    for url in req.image_urls:
        image = get_image_from_url(url)
        if image is None:
            continue
        score = compute_similarity_score(image, req.description)
        if score >= req.threshold:
            return {"url": url, "score": round(score, 2)}
    raise HTTPException(status_code=404, detail="No matching image found.")