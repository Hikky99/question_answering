from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
import pandas as pd
import numpy as np

app = Flask(__name__)

# Connect to ElasticSearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

# Load the SentenceTransformers model for passage embeddings
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Paths to input and output CSV files
input_csv_path = 'docs/passage_metadata.csv'
output_csv_path = 'docs/passage_metadata_emb.csv'

# Read the input CSV containing passages and metadata
data = pd.read_csv(input_csv_path)

@app.route('/')
def hello_world():
    return 'Welcome to the Question-Answering System API!'

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data['question']

        # Implement logic to retrieve relevant passages from ElasticSearch
       

        # Format the answers as needed
        formatted_answers = format_answers(answers)

        return jsonify({'answers': formatted_answers})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def format_answers(answers):
    # Implement logic to format the retrieved answers
   

if __name__ == '__main__':
    app.run(debug=True)
