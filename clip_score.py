from flask import Flask, request, jsonify
import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Helper to get image from URL
def get_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Error loading {url}: {e}")
        return None

# Compute similarity
def compute_similarity_score(image, text):
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        similarity = outputs.logits_per_image[0][0].item()
    score = (similarity / 30) * 100  # Normalize
    return score

# Find matching image
def find_matching_image_url(image_urls, description, threshold=80):
    for url in image_urls:
        image = get_image_from_url(url)
        if image is None:
            continue
        score = compute_similarity_score(image, description)
        if score >= threshold:
            return {"url": url, "score": score}
    return None

# Flask route
@app.route('/match-image', methods=['POST'])
def match_image():
    data = request.get_json()
    if not data or 'caption' not in data or 'image_url' not in data:
        return jsonify({'error': 'Invalid input. Requires "caption" and "image_url"'}), 400

    caption = data['caption']
    image_urls = data['image_url']

    if not isinstance(caption, str) or not isinstance(image_urls, list):
        return jsonify({'error': '"caption" must be a string and "image_url" must be a list'}), 400

    result = find_matching_image_url(image_urls, caption)
    if result:
        return jsonify(result), 200
    else:
        return jsonify({'error': 'No matching image found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
