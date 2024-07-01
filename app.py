import io
import base64
from flask import Flask, request, render_template
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from model import prepare_model
from utils import get_segment_labels, draw_segmentation_map, image_overlay
from config import ALL_CLASSES, LABEL_COLORS_LIST

app = Flask(__name__)

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = prepare_model(num_classes=3).to(device)  # Change num_classes to 3
checkpoint = torch.load('model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def apply_mask(image, mask):
    image = np.array(image)
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    for label_index, color in enumerate(LABEL_COLORS_LIST):
        colored_mask[mask_resized == label_index] = color
    return colored_mask

def predict_image(img, model):
    # Get the segment labels
    outputs = get_segment_labels(img, model, device)
    outputs = outputs['out']
    
    # Draw the segmentation map
    segmented_image = draw_segmentation_map(outputs)
    
    # Overlay the segmentation map on the original image
    final_image = image_overlay(img, segmented_image)
    
    # Convert the result image to base64 to display in HTML
    buffered = io.BytesIO()
    final_image_pil = Image.fromarray(final_image)
    final_image_pil.save(buffered, format="JPEG")
    img_data = base64.b64encode(buffered.getvalue()).decode()

    return img_data

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', img_data=None)
        file = request.files['file']
        if file:
            img = Image.open(file.stream).convert('RGB')
            results = predict_image(img, model)
            return render_template('index.html', img_data=results)
    return render_template('index.html', img_data=None)

if __name__ == '__main__':
    app.run(debug=True)
