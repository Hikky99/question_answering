# Question Answering System with SentenceTransformers and ElasticSearch

This project is a question-answering (QA) system that leverages SentenceTransformers models for generating embeddings, ElasticSearch as a vector store, Docker for containerization and deployment, and Flask for the API. The system can take a user's question, search for relevant passages in a given corpus, and return them as answers.

## Table of Contents


- [Usage](#usage)
- [Docker Containerization](#docker-containerization)
- [API Deployment](#api-deployment)
- [Evaluation](#evaluation)
- [License](#license)



# Usage
### Parsing
Use the parsing.py script to parse legal files and generate passage_metadata.csv.

### Passage Embeddings
Generate passage embeddings using SentenceTransformers and save them in passage_metadata_emb.csv using the model.py script.

### ElasticSearch Integration
Set up an ElasticSearch index and index the data using the indexing.py script.

### Document Retrieval
Implement the retrieval logic in the retrieval.py script to retrieve relevant passages from ElasticSearch.

### API Deployment
Create a Flask API using the app.py script for user interaction.

### Evaluation
Evaluate the system using manually derived evaluation data.



# Docker Containerization
Containerize the application using Docker for easy deployment. Refer to Dockerfile and docker-compose.yml for configuration.

## Build and run the Docker container

docker-compose up


# API Deployment
The Flask API exposes endpoints for receiving user questions and uploading documents to be indexed. Refer to app.py for the API implementation.

# Run the Flask API
python app.py

# Evaluation
Evaluate the performance of the system manually. 



# License

