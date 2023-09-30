from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# Load the SentenceTransformers model (you can replace 'bert-base-nli-mean-tokens' with your preferred model)
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Paths to input and output CSV files
input_csv_path = 'docs/passage_metadata.csv'
output_csv_path = 'docs/passage_metadata_emb.csv'

# Read the input CSV containing passages and metadata
data = pd.read_csv(input_csv_path)

# Initialize lists to store passages, metadata, and embeddings
passages = []
metadata = []
embeddings = []

# Track progress
total_passages = len(data)
processed_passages = 0

# Iterate through rows of the input CSV
for index, row in data.iterrows():
    passage = row['Passage']

    # Generate embeddings for the passage using the model
    passage_embedding = model.encode(passage, convert_to_tensor=True)

    # Check if passage_embedding has a value
    if passage_embedding is not None:
        # Append the passage and metadata to their respective lists
        passages.append(passage)
        metadata.append(row['Metadata'])

        # Append the entire embedding vector to the embeddings list
        embeddings.append(passage_embedding.tolist())
    else:
        # Handle the case where passage_embedding is None
        # You can choose to skip this passage or handle it differently
        print(f"Warning: passage_embedding is None for passage at index {index}")

    # Update progress
    processed_passages += 1
    print(f"Processed {processed_passages}/{total_passages} passages")

# Check if any embeddings were generated
if embeddings:
    # Create a DataFrame for the output data
    output_data = pd.DataFrame({'Passage': passages, 'Metadata': metadata, 'Embedding': embeddings})

    # Save the DataFrame to the output CSV
    output_data.to_csv(output_csv_path, index=False)
else:
    print("No valid embeddings were generated.")
