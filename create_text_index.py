# create_text_index.py

from elasticsearch import Elasticsearch
import json

def create_text_vector_index():
    # Initialize Elasticsearch client
    es = Elasticsearch(hosts=["http://localhost:9200"])

    # Define the mapping for the text vector index
    mapping = {
        "mappings": {
            "properties": {
                "text_vector": {
                    "type": "dense_vector",
                    "dims": 768,
                    "index": True,
                    "similarity": "cosine"
                },
                "text_id": {
                    "type": "keyword"
                }
            }
        }
    }

    # Check if the index already exists
    if not es.indices.exists(index='text_vector_index_beta'):
        # Create the index with the defined mapping
        es.indices.create(index='text_vector_index_beta', body=mapping)
        print("Index 'text_vector_index_beta' created successfully!")
    else:
        print("Index 'text_vector_index_beta' already exists!")

if __name__ == "__main__":
    create_text_vector_index()