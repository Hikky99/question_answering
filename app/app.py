from flask import Flask, request, jsonify, render_template_string
from elasticsearch import Elasticsearch
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from sentence_transformers import SentenceTransformer



app = Flask(__name__)

app.secret_key = 'your_secret_key_here'


# Connect to Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# Load the SentenceTransformers model for passage embeddings
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Path to the input CSV file containing passage metadata and embeddings
input_csv_path = 'docs/passage_metadata_emb.csv'

# Create a Flask-WTF form for the question input
class QuestionForm(FlaskForm):
    question = StringField('Enter your question:')
    submit = SubmitField('Submit')

@app.route('/')
def index():
    form = QuestionForm()
    return render_template_string("""
<!DOCTYPE html>
<link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='styles.css') }}">

<html>
<head>
    <title>Question-Answering System</title>
</head>
<body>
    <h1>Question-Answering System</h1>
    <form method="POST" action="/ask">
        {{ form.hidden_tag() }}
        <label for="question">Enter your question:</label>
        {{ form.question() }}
        <br>
        {{ form.submit() }}
    </form>
</body>
</html>
""", form=form)

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

        return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>Question-Answering System</title>
</head>
<body>
    <h1>Question-Answering System</h1>
    <form method="POST" action="/ask">
        {{ form.hidden_tag() }}
        <label for="question">Enter your question:</label>
        {{ form.question() }}
        <br>
        {{ form.submit() }}
    </form>
    <h2>Results</h2>
    <p>Question: {{ question }}</p>
    <ul>
    {% for answer in answers %}
        <li>{{ answer }}</li>
    {% endfor %}
    </ul>
</body>
</html>
""", question=question, answers=relevant_passages)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
