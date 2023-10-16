# Advanced Image Retrieval System

Harness the combined power of Elasticsearch, deep learning models like ResNet50 and BERT, and YOLOv8 for object detection to offer a multi-dimensional image retrieval solution. This project provides diverse retrieval methods, from text-based and semantic search to feature vector and object-based techniques, ensuring comprehensive and relevant image search results.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Key Optimizations](#key-optimizations)
5. [Setup and Installation](#setup-and-installation)
6. [Usage](#usage)
7. [Contribution](#contribution)
8. [License](#license)

## Introduction

The digital age has seen an exponential growth in image data. Efficient retrieval techniques that go beyond just text-based search are the need of the hour. This project addresses this challenge, offering a multi-pronged approach to image retrieval.

## Features

### Text-Based Search
Leverage the efficiency of Elasticsearch for vast datasets using textual queries.

### Image-Based Search
Utilize ResNet50 for feature extraction, enabling the system to compare and find visually similar images using ElasticKNN.

### Semantic Search
- Benefit from the powerful BERT model, which transforms image titles into embeddings.
- Given a textual query, derive its embedding and retrieve semantically relevant images.

### Object-Based Search with YOLOv8
- Identify objects in uploaded images using the state-of-the-art YOLOv8 model.
- Users can then choose an object from the detected list, and the system retrieves images containing the selected object.

### Index Creation
Detailed steps to create your own index are provided in a separate README file "Setting Up and Using Elasticsearch, Logstash, and Kibana (ELK Stack) on Ubuntu.md". Ensure you go through it if you're looking to customize or set up a new index for your use-case.
## Project Structure

- `app.py`: The primary Streamlit application driving the user interface and interactions with Elasticsearch.
- `config.py`: Centralized configuration settings for the application.
- `es_operations.py`: Dedicated functions for Elasticsearch operations.
- `image_validation.log`: Log file capturing details about image validation processes.
- `image_validator.py`: Logic to validate images before indexing to Elasticsearch.
- `logger_setup.py`: Configurations for setting up logging for the application.
- `main.py`: The main execution script for the application.
- `choose_model.py`: Setup for the BERT model used in semantic search.
- `create_text_index.py`: Scripts for creating an index tailored for semantic search.
- `feature_extraction.py`: Leverage ResNet50 to extract image features.
- `generate_embeddings.py`: Convert text into embeddings using BERT.
- `index_images.py`: Handle the indexing of images.
- `index_text_embeddings.py`: Scripts for indexing text embeddings.
- `knn_mapping.json`: Provides the specific mapping required for image indexing compatible with ElasticKNN.
- `yolo.py`: Object detection in images using the YOLOv8 model.

## Key Optimizations

### Pre-Validation with `is_valid` Flag
Ensure relevance and speed in search results by pre-validating images. Marking images with this flag allows Elasticsearch to prioritize and return valid images, optimizing user search experience.

### Efficient Data Handling with Scroll API
Streamline the retrieval of vast datasets from Elasticsearch without manual pagination, ensuring consistent and fast results even with underlying data changes.

## Setup and Installation

1. Ensure Python 3.x is installed.
2. Clone the repository:
    ```bash
    git clone https://github.com/Houssem-Ben-Salem/Advanced-Image-Retrieval-System.git
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

   **Note**: Ensure Elasticsearch with ElasticKNN and YOLOv8 are set up and operational.

4. Set your Elasticsearch parameters in `config.py`.

5. Launch the application:
    ```bash
    streamlit run app.py
    ```

## Usage

- Begin with `app.py` using Streamlit to experience the versatile retrieval interface.
- Follow on-screen instructions to choose a retrieval method and provide the necessary input.

## Contribution

For significant modifications, open an issue for discussion or directly submit a pull request. All contributions are welcome.

## License

[MIT](https://choosealicense.com/licenses/mit/)
