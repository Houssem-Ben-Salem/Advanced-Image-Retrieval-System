from elasticsearch import Elasticsearch
import csv

def index_feature_vectors_to_elasticsearch(csv_file_path):
    es = Elasticsearch(hosts=["http://localhost:9200"])

    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # skip header row
        
        for row in reader:
            # Default behavior
            image_id, vector_str = row[0], row[1]
            vector = [float(i) for i in vector_str.split(',')]
            
            # If vector length is not 2048, adjust parsing
            if len(vector) != 2048:
                image_id, vector_str = row[0], row[1:]
                vector = [float(i) for i in vector_str]
            
            body = {
                "image_id": image_id,
                "image_vector": vector
            }
            
            es.index(index="image_vector_index_beta", body=body)

    print("Indexing completed!")

if __name__ == "__main__":
    csv_file_path = '/home/hous/Desktop/flask_image_search/combined_features.csv'
    index_feature_vectors_to_elasticsearch(csv_file_path)
