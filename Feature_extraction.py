import torch
import timm
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import csv
import requests
from io import BytesIO

"""
# Load pre-trained EfficientNet model and set to evaluation mode
model_name = 'tf_efficientnet_b0_ns'  # You can change this to other variants if needed
effnet = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='')
effnet.eval()
"""
# --- Initialization ---

# Initialize the ResNet50 model
resnet50 = models.resnet50(pretrained=True)
resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-1]))  # Remove the last FC layer
resnet50.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features_from_url(url):
    """Extract feature vector from an image URL."""
    response = requests.get(url)

    # Check if the URL fetch was successful
    if response.status_code != 200:
        print(f"Failed to fetch {url}. Status code: {response.status_code}")
        return None

    img = Image.open(BytesIO(response.content))

    # Check if the image has valid dimensions
    if img.size[0] <= 0 or img.size[1] <= 0:
        print(f"Invalid image dimensions for {url}.")
        return None

    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = resnet50(img)
    # Squeeze to get rid of unnecessary dimensions and convert tensor to list
    return features.squeeze().tolist()

def construct_image_url(row):
    """Constructs the image URL from a row of metadata."""
    return f"http://farm{row['flickr_farm']}.staticflickr.com/{row['flickr_server']}/{row['id']}_{row['flickr_secret']}.jpg"

START_ID = '4227424460'  # Specify the ID from where you want to start

if __name__ == "__main__":
    input_csv = "photo_metadata.csv"
    output_csv = "extracted_feature.csv"
    should_start_processing = False

    with open(input_csv, "r") as infile, open(output_csv, "a", newline='') as outfile:  # Open outfile in append mode
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)

        if START_ID == '':  # If START_ID is an empty string, start processing from the beginning
            should_start_processing = True
            writer.writerow(["id", "image_vector"])  # Add header only when starting afresh

        for row in reader:
            # Check if current row's ID matches the start ID
            if row['id'] == START_ID:
                should_start_processing = True
                continue  # Skip the image with the start ID and process from the next one

            # If we haven't reached the start ID, skip the current iteration
            if not should_start_processing:
                continue

            image_url = construct_image_url(row)
            try:
                feature_vector = extract_features_from_url(image_url)
                if feature_vector:  # Only proceed if feature_vector is not None
                    print(type(feature_vector))  # It should print: <class 'list'>
                    print(len(feature_vector))  # This will print the length of the feature vector
                    writer.writerow([row['id']] + feature_vector)
            except Exception as e:
                print(f"Error processing {row['id']}: {e}")
