import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Load the BERT model once at startup
print("Loading BERT model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded!")


@app.route("/rank", methods=["POST"])
def rank_results():
    data = request.json
    query = data.get("query")
    documents = data.get("documents", [])

    if not query or not documents:
        return jsonify({"error": "Missing query or documents"}), 400

    # Extract text for ranking (title + snippet)
    doc_texts = [f"{doc['title']} {doc['snippet']}" for doc in documents]

    # Encode query and documents
    query_embedding = model.encode(query, convert_to_tensor=True)
    doc_embeddings = model.encode(doc_texts, convert_to_tensor=True)

    # Calculate cosine similarity
    scores = util.cos_sim(query_embedding, doc_embeddings)[0]

    # Combine documents with their scores
    results = []
    for doc, score in zip(documents, scores):
        results.append(
            {
                "title": doc["title"],
                "snippet": doc["snippet"],
                "pageid": doc["pageid"],
                "score": float(score),
            }
        )

    # Sort by score (highest first)
    results.sort(key=lambda x: x["score"], reverse=True)

    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
