import torch
import numpy as np
from torchvision import transforms
import cv2
from config import label_color_map

def get_segment_labels(image, model, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
    return {'out': outputs['out']}

def draw_segmentation_map(outputs):
    labels = torch.argmax(outputs.squeeze(), dim=0).cpu().numpy()
    segmented_image = np.zeros((256, 256, 3), dtype=np.uint8)
    for label in np.unique(labels):
        if label == 0:
            continue
        mask = labels == label
        color = label_color_map[label]
        segmented_image[mask] = color
    return segmented_image

def image_overlay(image, segmented_image):
    image = np.array(image)
    segmented_image_resized = cv2.resize(segmented_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(image, 0.5, segmented_image_resized, 0.5, 0)
