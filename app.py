from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import boto3
import os
from io import BytesIO
from torch import nn


'''
Before you run the app.py, do the following, to set variabes for access keys and secret key.

'export AWS_ACCESS_KEY_ID equal to xxx'g
'export AWS_SECRET_ACCESS_KEY equal to xxx'

This is to adhere to security practices
'''

# Initialize Flask app
app = Flask(__name__)

# Initialize Boto3 S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name='eu-west-2'
)

bucket_name = 'oritsejolomi-fyp'

# Load the precomputed code embeddings and raw code snippets from S3 bucket
code_embeddings = None
combined_raw_code = None
with BytesIO() as data:
    s3_client.download_fileobj(bucket_name, 'code_embeddings.npy', data)
    data.seek(0)
    code_embeddings = np.load(data, allow_pickle=True)

    s3_client.download_fileobj(bucket_name, 'combined_raw_code.pkl', data)
    data.seek(0)
    combined_raw_code = pickle.load(data)
    # using s3_client.download_fileobj can be more memory-efficient since it allows to read contents 
    # of S3 objects directly into memory without storing them on disk,

# Load the pre-trained model
model = SentenceTransformer('model_directory')

#Define cosine_similarity to be accessible everywhere 
cosine_similarity = nn.CosineSimilarity(dim=1)

@app.route('/')
def root():
    return render_template('query.html')

# Define an endpoint to handle queries
@app.route("/query", methods=['POST'])
def query():
    # Parse JSON data from request
    data = request.json
    user_query = data['query']
    user_language = data['language']

    # Log the received query and language
    app.logger.info(f"Received query: {user_query}, Language: {user_language}")

    app.logger.info("Generating embedding for user query...")
    query_with_label = f"{user_query} [LANG] {user_language}"

    # Generate embedding for user query
    query_embedding = model.encode([query_with_label])[0]

    app.logger.info("Query embedded, now creating tensor...")

    #query_embedding_tensor = torch.tensor(query_embedding).unsqueeze(0)
    
    app.logger.info("Tensor Created")

    # Compute cosine similarity
    app.logger.info("Now Performing Cosine Similarity")
    cos = cosine_similarity(torch.tensor(query_embedding).unsqueeze(0), torch.tensor(code_embeddings))

    # Get the most similar code snippets
    top_k = 5  # Adjust based on how many results you want
    closest_n = torch.argsort(cos, descending=True)[:top_k]
    results = [combined_raw_code[idx] for idx in closest_n]

    app.logger.info("Results generated")
    # Return results as JSON response
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
