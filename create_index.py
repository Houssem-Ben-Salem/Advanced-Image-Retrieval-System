# create_index.py

from elasticsearch import Elasticsearch
import json

def create_image_vector_index():
    # Initialize Elasticsearch client
    es = Elasticsearch(hosts=["http://localhost:9200"])

    # Load the mapping from the JSON file
    with open('knn_mapping.json', 'r') as file:
        mapping = json.load(file)

    # Check if the index already exists
    if not es.indices.exists(index='image_vector_index_beta'):
        # Create the index with the loaded mapping
        es.indices.create(index='image_vector_index_beta', body=mapping)
        print("Index 'image_vector_index_beta' created successfully!")
    else:
        print("Index 'image_vector_index_beta' already exists!")

if __name__ == "__main__":
    create_image_vector_index()
