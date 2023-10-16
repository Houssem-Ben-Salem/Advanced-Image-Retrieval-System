# index_text_embeddings.py

from elasticsearch import Elasticsearch
import csv

def index_text_embeddings_to_elasticsearch(csv_file_path, index_name="text_vector_index_beta"):
    es = Elasticsearch(hosts=["http://localhost:9200"])
    
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # skip header row

        count = 0
        for row in reader:
            text_id, embedding_str = row[0], row[1]
            embedding_str = embedding_str.strip().replace('\n', ' ').replace('[', '').replace(']', '')
            embedding = [float(i) for i in embedding_str.split()]

            
            body = {
                "text_id": text_id,
                "text_vector": embedding
            }
            
            es.index(index=index_name, body=body)
            count += 1

            if count % 1000 == 0:  # print every 1000 indexed rows
                print(f"Indexed {count} lines")

    print("Indexing completed!")

if __name__ == "__main__":
    csv_file_path = 'embeddings.csv'
    index_text_embeddings_to_elasticsearch(csv_file_path)
