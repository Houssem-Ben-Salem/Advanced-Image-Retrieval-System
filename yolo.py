import torch
import requests
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO
from io import BytesIO

# Load YOLO model
model = YOLO('yolov8n.pt')

# Initialize the ResNet50 model
resnet50 = models.resnet50(pretrained=True)
resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-1]))  # Remove the last FC layer
resnet50.eval()

# Image transformations for ResNet50
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(img_path):
    """Extract feature vector from an uploaded image."""
    img = Image.open(img_path)
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    with torch.no_grad():
        features = resnet50(batch_t)
    return features.squeeze().tolist()

def fetch_and_extract_features_from_stream(uploaded_stream):
    """Fetch image from a given stream and extract feature vectors for each detected bounding box."""
    
    img = Image.open(uploaded_stream)
    img.save('temp_image.jpg')  # Save the uploaded image for processing
    
    # Detect bounding boxes in the image
    results = model.predict('temp_image.jpg', save=False, imgsz=320, conf=0.7)
    
    feature_vectors = []
    detected_objects = []  # This will store the detected object labels

    # Assuming YOLO's class names are available
    class_names = model.names

    # Access the first result (assuming it contains the required attributes)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()  # If this fails, check if the attribute exists in the printed output

    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = map(int, box)
        
        cropped_img = img.crop((xmin, ymin, xmax, ymax))
        cropped_img.save('temp_cropped.jpg')  # Save the cropped region for feature extraction
        feature = extract_features('temp_cropped.jpg')
        feature_vectors.append(feature)

        # Add the detected object's label based on its class number
        detected_objects.append(class_names[int(classes[i])])
    
    return feature_vectors, detected_objects
