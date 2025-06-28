from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import subprocess

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Global store for chunks and index
chunks = []
index = None

@app.route("/")
def home():
    return render_template("index2.html")

@app.route("/upload", methods=["POST"])
def upload():
    global chunks, index

    file = request.files.get("pdf")
    if not file:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Extract text
    reader = PyPDF2.PdfReader(filepath)
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

    # Chunk text into 500-character segments
    chunks = [full_text[i:i+500] for i in range(0, len(full_text), 500)]

    # Create embeddings
    vectors = embedder.encode(chunks)
    vectors = np.array(vectors).astype("float32")

    # Build FAISS index
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    return jsonify({"status": "success", "message": f"✅ {len(chunks)} chunks indexed from PDF!"})

@app.route("/ask", methods=["POST"])
def ask():
    global chunks, index

    data = request.get_json()
    query = data.get("message", "").strip()

    if not query:
        return jsonify({"response": "❗No query provided."}), 400
    if not index or not chunks:
        return jsonify({"response": "⚠️ Please upload a PDF file first."}), 400

    try:
        query_vec = embedder.encode([query]).astype("float32")
        D, I = index.search(query_vec, 5)
        relevant_chunks = "\n".join([chunks[i] for i in I[0]])

        # Build the prompt
        prompt = f"Context:\n{relevant_chunks}\n\nQuestion: {query}\nAnswer:"

        # Run with Ollama
        result = subprocess.run(
            ["ollama", "run", "llama3:8b", prompt],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            raise Exception(result.stderr.strip())

        return jsonify({"response": result.stdout.strip()})

    except Exception as e:
        return jsonify({"response": f"⚠️ Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
