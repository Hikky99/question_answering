from elasticsearch import Elasticsearch
import sys
import pandas as pd
import numpy as np

# Paths to input CSV files
input_csv_path = '../docs/passage_metadata_emb.csv'


# Connect to your ElasticSearch cluster
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])





# Check if the ElasticSearch index already exists, if not, create it
index_name = 'passage_embeddings'  # You can choose a suitable name
if not es.indices.exists(index=index_name):
    # Define the mapping for your index (customize as needed)
    mapping = {
        "mappings": {
            "properties": {
                "Passage": {"type": "text"},
                "Metadata": {"type": "keyword"},
                "Embedding": {"type": "dense_vector", "dims": 768}
            }
        }
    }

    # Create the index with the specified mapping
    es.indices.create(index=index_name, body=mapping)
    print(f"Index '{index_name}' created successfully.")

# Read the input CSV containing passages, metadata, and embeddings
data = pd.read_csv("../docs/passage_metadata_emb.csv")

# Index each row of data into ElasticSearch
for index, row in data.iterrows():
    doc = {
        'Passage': row['Passage'],
        'Metadata': row['Metadata'],
        'Embedding': np.fromstring(row['Embedding'], sep=',')  # Convert the string back to a NumPy array
    }

    # Index the document into the ElasticSearch index
    es.index(index=index_name, document=doc)

print(f"Indexed {len(data)} documents into '{index_name}' index.")
