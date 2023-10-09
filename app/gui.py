from flask import Flask, request, jsonify, render_template
from elasticsearch import Elasticsearch
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your secret key

# Connect to ElasticSearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# Load the SentenceTransformers model for passage embeddings
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Paths to input and output CSV files
input_csv_path = 'docs/passage_metadata_emb.csv'

# Create a Flask-WTF form for the question input
class QuestionForm(FlaskForm):
    question = StringField('Enter your question:')
    submit = SubmitField('Submit')

@app.route('/')
def index():
    form = QuestionForm()
    return render_template('index.html', form=form)

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.form
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

        return render_template('results.html', question=question, answers=relevant_passages)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
