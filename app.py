from flask import Flask, request, jsonify, render_template
import subprocess

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("message", "").strip()

    if not query:
        return jsonify({"response": "❗No query provided."}), 400

    try:
        # Run Ollama with the locally available model
        result = subprocess.run(
            ["ollama", "run", "llama3:8b", query],
            capture_output=True,
            text=True,
            timeout=60
        )

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        if result.returncode != 0:
            raise Exception(result.stderr.strip())

        return jsonify({"response": result.stdout.strip()})

    except Exception as e:
        print("Exception occurred:", str(e))  # 🐞 Debug print
        return jsonify({"response": f"⚠️ Error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
