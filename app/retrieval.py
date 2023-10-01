from elasticsearch import Elasticsearch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Initialize ElasticSearch connection
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

# Define the ElasticSearch index name
index_name = 'passage_embeddings'

def retrieve_top_passages(query, num_results=3):
    """
    Retrieve the top relevant passages from ElasticSearch based on the user's query.

    Args:
        query (str): The user's question/query.
        num_results (int): The number of top relevant passages to retrieve.

    Returns:
        list: A list of dictionaries containing relevant passages and their metadata.
    """
    # Perform a query to retrieve relevant documents
    search_body = {
        "query": {
            "match": {
                "Passage": query
            }
        },
        "size": num_results
    }

    # Execute the search
    search_results = es.search(index=index_name, body=search_body)

    # Extract and format the relevant passages and metadata
    relevant_documents = []
    for hit in search_results['hits']['hits']:
        passage = hit['_source']['Passage']
        metadata = hit['_source']['Metadata']
        embedding = np.fromstring(hit['_source']['Embedding'], sep=',')
        relevance_score = cosine_similarity([embedding], [query_embedding])[0][0]

        relevant_document = {
            "Passage": passage,
            "Metadata": metadata,
            "Relevance Score": relevance_score
        }

        relevant_documents.append(relevant_document)

    # Sort by relevance score in descending order
    relevant_documents.sort(key=lambda x: x["Relevance Score"], reverse=True)

    return relevant_documents[:num_results]

if __name__ == "__main__":
    # Example query
    user_query = "What are the principles of an injunction?"

    # Number of top passages to retrieve
    num_results_to_retrieve = 3

    # Generate an example query embedding (you should use embeddings from your model)
    query_embedding = np.random.rand(768)

    # Retrieve relevant passages
    relevant_passages = retrieve_top_passages(user_query, num_results_to_retrieve)

    # Create a DataFrame for the results
    result_df = pd.DataFrame(relevant_passages)

    # Save the results to a CSV file
    result_df.to_csv('questions_answers.csv', index=False)
