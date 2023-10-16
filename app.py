import streamlit as st
from elasticsearch import Elasticsearch
from googletrans import Translator
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sample_and_embedding import preprocess, get_embedding
from transformers import BertTokenizer, BertModel
import torch
from yolo import fetch_and_extract_features_from_stream
from io import BytesIO
import base64
import speech_recognition as sr


# --- Elasticsearch Setup ---
es = Elasticsearch(hosts=["http://localhost:9200"])

def get_image_download_link(img, filename, format):
    buffered = BytesIO()
    img.save(buffered, format=format.upper())
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/{format.lower()};base64,{img_str}" download="{filename}">Download {filename}</a>'
    return href

# --- Initialization ---

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


def search_by_feature_vector(vector):
    body = {
        "knn": {
            "field": "image_vector",  # Note: In the example, you used "image-vector". Ensure the field name matches the one in your mapping.
            "query_vector": vector,
            "k": 10,
            "num_candidates": 100
        },
        "_source": ["image_id"]  # Replace "title" and "file-type" with fields you need. I kept "image_id" here based on your initial code.
    }
    response = es.search(index='image_vector_index_beta', body=body)
    return response


def get_image_by_id(image_id):
    body = {
        "query": {
            "term": {
                "id": image_id
            }
        }
    }
    response = es.search(index='flickrdata', body=body)
    if response['hits']['hits']:
        return response['hits']['hits'][0]['_source']['image_url']
    return None
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def get_text_embedding(text):
    """Generates a BERT embedding for the given text after preprocessing."""
    preprocessed_text = preprocess(text)
    embedding = get_embedding(preprocessed_text, tokenizer, model)
    return embedding

def search_by_text_embedding(query):
    body = {
        "knn": {
            "field": "text_vector",  # Use the field name where you store text embeddings
            "query_vector": query,
            "k": 10,
            "num_candidates": 100
        },
        "_source": ["text_id"]
    }
    response = es.search(index='text_vector_index_beta', body=body)  # Use the correct index for text embeddings
    return response

def search_elasticsearch(query, language, size=5, start_from=0):
    if language != "English":
        # Translate the query to English
        translated = translator.translate(query, src=language.lower(), dest='en')
        query = translated.text

    body = {
        "query": {
            "bool": {
                "must": {
                    "match": {
                        "tags": query
                    }
                },
                "filter": {
                    "term": {"is_valid": True}
                }
            }
        },
        "size": size,
        "from": start_from
    }
    response = es.search(index='flickrdata', body=body)
    return response
# Initialize the speech recognizer
recognizer = sr.Recognizer()

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        body {{
            color: white;  # Setting text color to white
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

st.set_page_config(layout="wide")
set_background('background.png')


translator = Translator()

# Sidebar for page navigation
st.sidebar.header('Navigation')
page = st.sidebar.radio('Select a page:', ('Main Search', 'Semantic Search', 'Object-based Image Retrieval'))

# --- App Title ---
st.markdown("<h1 style='color: white; text-align: center;'>Flickr Image Search 🖼️</h1>", unsafe_allow_html=True)

st.markdown("<div style='color: white; text-align: center;'>Welcome to Flickr Image Search! Type in tags and find relevant images. Let's get started!</div>", unsafe_allow_html=True)

# Add vertical spacing
st.markdown("<br><br>", unsafe_allow_html=True)

def main_search():
        
    with st.container():
        col_space1, col1, col_space2, col2, col_space3, col3, col_space4, col4, col_space5, col5, col_space6 = st.columns((1,2,1,2,1,2,1,2,1,2,1))

        with col1:
            query_placeholder = "<div style='color: white;'>Search for images by tags:</div>"
            st.markdown(query_placeholder, unsafe_allow_html=True)
            query = st.text_input('', '')

        with col2:
            language_label = "<div style='color: white;'>Choose language:</div>"
            st.markdown(language_label, unsafe_allow_html=True)
            language = st.selectbox("", ["English", "French", "Arabic", "Spanish", "German"])

        with col3:
            num_images_label = "<div style='color: white;'>N° Images to Display</div>"
            st.markdown(num_images_label, unsafe_allow_html=True)
            num_images = st.number_input('', min_value=1, step=5, value=5)

        with col4:
            st.markdown("<div style='height: 58px;'></div>", unsafe_allow_html=True)  # spacer to align with other columns
            search_button = st.button('Search')


        with col5:  # Voice search button alignment
            st.markdown("<div style='height: 58px;'></div>", unsafe_allow_html=True)  # spacer to align with other columns
            voice_search_button = st.button("Voice Search")

    if voice_search_button:
        query = None
        with sr.Microphone() as source:
            st.write("Listening for your query...")
            try:
                audio = recognizer.listen(source, timeout=5)
                query = recognizer.recognize_google(audio)
            except sr.WaitTimeoutError:
                st.warning("Listening timeout. Please try again.")
            except sr.UnknownValueError:
                st.warning("Sorry, I could not understand your query. Please try again.")

        if query:
            st.text(f"You said: {query}")

    if search_button or query:
        if query:
            with st.spinner('Searching...'):
                results = search_elasticsearch(query, language, size=num_images)
                hits = results.get('hits', {}).get('hits', [])
                    
            if not hits:
                st.warning('No matching images found.')  # Use warning style to make it stand out
            else:
                # Display images
                for i in range(0, len(hits), 3):
                    row_images = [hit['_source']['image_url'] for hit in hits[i:i + 3]]
                    image_row = st.columns(3)
                    for j, image_url in enumerate(row_images):
                        with image_row[j]:
                            st.image(image_url, caption=f'Image {i + j + 1}', use_column_width=True)
    # Feature extraction and similarity search (to be implemented)
    uploaded_image = st.file_uploader("Upload an image to find similar images:", type=['jpg', 'jpeg', 'png'])

    if uploaded_image:
        with st.spinner('Extracting features...'):
            vector = extract_features(uploaded_image)
        with st.spinner('Searching...'):
            results = search_by_feature_vector(vector)
            hits = results.get('hits', {}).get('hits', [])
        if not hits:
            st.info('No matching images found.')
        else:
            # Display the uploaded image
            st.header("Uploaded Image:")
            st.image(uploaded_image, caption="Your uploaded image", use_column_width=True)

            st.header("Similar Images:")

            # Display images
            image_urls = [get_image_by_id(hit['_source']['image_id']) for hit in hits if get_image_by_id(hit['_source']['image_id'])]
            
            for i in range(0, len(image_urls), 2):
                row_images = image_urls[i:i + 2]
                image_row = st.columns(2)
                for j, image_url in enumerate(row_images):
                    with image_row[j]:
                        st.image(image_url, caption=f'Image ID: {hits[i + j]["_source"]["image_id"]}', use_column_width=True)
def semantic_search():
    """Handles the semantic search UI and functionality."""
    st.markdown("<h2 style='color: white; text-align: center;'>Semantic Image Search 🧠</h2>", unsafe_allow_html=True)
    
    query = st.text_input('Enter your semantic query:', '')
    search_button = st.button('Search Semantically')
    
    if search_button:
        # Obtain BERT embeddings for the query
        query_vector = get_text_embedding(query)
        
        # Perform the search using the embeddings
        with st.spinner('Searching semantically...'):
            results = search_by_text_embedding(query_vector)
            hits = results.get('hits', {}).get('hits', [])
            
        if not hits:
            st.warning('No matching images found.') 
        else:
            # Retrieve and display images
            image_urls = [get_image_by_id(hit['_source']['text_id']) for hit in hits if get_image_by_id(hit['_source']['text_id'])]
            for i in range(0, len(image_urls), 2):
                row_images = image_urls[i:i + 2]
                image_row = st.columns(2)
                for j, image_url in enumerate(row_images):
                    with image_row[j]:
                        st.image(image_url, caption=f'Image ID: {hits[i + j]["_source"]["text_id"]}', use_column_width=True)

def object_based_retrieval():
    """
    Handles the object-based image retrieval UI and functionality.
    """
    st.markdown("<h2 style='color: white; text-align: center;'>Object-based Image Retrieval 📸</h2>", unsafe_allow_html=True)
    
    uploaded_image = st.file_uploader("Upload an image to detect objects and find similar images:", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_image:
        # Display the uploaded image
        st.image(uploaded_image, use_column_width=True)
        st.markdown("<p style='color: white;'>Uploaded Image</p>", unsafe_allow_html=True)

        # Save the uploaded image for processing
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_image.getbuffer())
        
        # Extract features for detected objects
        with st.spinner("Detecting objects and extracting features..."):
            feature_vectors, object_labels = fetch_and_extract_features_from_stream("uploaded_image.jpg") # Assume this function now also returns object labels

        # Construct and display a detected objects sentence
        detected_objects_str = ", ".join(object_labels[:-1]) + " and " + object_labels[-1] if len(object_labels) > 1 else object_labels[0]
        st.markdown(f"<p style='color: white;'>{len(object_labels)} objects: {detected_objects_str} were detected.</p>", unsafe_allow_html=True)

        # Display detected objects and allow the user to select which ones they want
        selected_objects = st.multiselect("Objects Detected:", object_labels, format_func=lambda x: f"{x}")

        # If no objects were selected by the user, show a message
        if not selected_objects:
            st.markdown("<p style='color: white;'>Please select objects to retrieve similar images.</p>", unsafe_allow_html=True)
        else:
            # For each selected object, perform a search and display relevant images
            for index, label in enumerate(object_labels):
                if label in selected_objects:
                    with st.spinner(f'Searching for similar images to "{label}"...'):
                        results = search_by_feature_vector(feature_vectors[index])
                        hits = results.get('hits', {}).get('hits', [])
                    
                    if not hits:
                        st.markdown(f"<p style='color: white;'>No matching images found for '{label}'.</p>", unsafe_allow_html=True)
                    else:
                        # Display images
                        st.markdown(f"<h3 style='color: white;'>Results for {label}:</h3>", unsafe_allow_html=True)
                        image_urls = [get_image_by_id(hit['_source']['image_id']) for hit in hits if get_image_by_id(hit['_source']['image_id'])]
                        for i in range(0, len(image_urls), 2):
                            row_images = image_urls[i:i + 2]
                            image_row = st.columns(2)
                            for j, image_url in enumerate(row_images):
                                with image_row[j]:
                                    st.image(image_url, use_column_width=True)
                                    st.markdown(f"<p style='color: white;'>Image ID: {hits[i + j]['_source']['image_id']}</p>", unsafe_allow_html=True)

# Updated routing based on page selection
if page == 'Main Search':
    main_search()
elif page == 'Semantic Search':
    semantic_search()
elif page == 'Object-based Image Retrieval':
    object_based_retrieval()