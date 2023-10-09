from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
import pandas as pd
import numpy as np

app = Flask(__name__)

# Connect to ElasticSearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# Load the SentenceTransformers model for passage embeddings
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Paths to input and output CSV files
input_csv_path = 'docs/passage_metadata_emb.csv'


@app.route('/')
def hello_world():
    return 'Welcome to the Question-Answering System API!'


@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data['question']

        # Define the Elasticsearch query based on the user's question
        query = {
            "query": {
                "match": {
                    "Passage": question
                }
            }
        }

        # Execute the search
        results = es.search(index=input_csv_path, body=query)

        # Extract relevant passages from the search results
        relevant_passages = []
        for hit in results['hits']['hits']:
            passage = hit['_source']['Passage']
            relevant_passages.append(passage)

        return jsonify({'answers': relevant_passages})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)