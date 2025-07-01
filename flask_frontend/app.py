from flask import Flask, render_template, request, redirect, url_for, jsonify
import requests

app = Flask(__name__)

BASE_URL = "http://localhost:8000"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/rewrite", methods=["GET", "POST"])
def rewrite():
    if request.method == "POST":
        original_text = request.form.get("original_text")
        target_tone = request.form.get("target_tone")
        target_platforms = request.form.getlist("target_platforms")
        brand_context = request.form.get("brand_context")
        target_audience = request.form.get("target_audience")

        data = {
            "original_text": original_text,
            "target_tone": target_tone,
            "target_platforms": target_platforms,
            "brand_context": brand_context,
            "target_audience": target_audience
        }

        try:
            response = requests.post(f"{BASE_URL}/run-agent", json=data)
            response.raise_for_status()
            result = response.json()
            return render_template("rewrite.html", result=result)
        except requests.exceptions.RequestException as e:
            return render_template("rewrite.html", error=str(e))

    return render_template("rewrite.html")

@app.route("/health")
def health():
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        health_status = response.json()
        return render_template("health.html", health_status=health_status)
    except requests.exceptions.RequestException as e:
        return render_template("health.html", error=str(e))

@app.route("/stats")
def stats():
    try:
        response = requests.get(f"{BASE_URL}/agent-stats")
        response.raise_for_status()
        stats = response.json()
        return render_template("stats.html", stats=stats)
    except requests.exceptions.RequestException as e:
        return render_template("stats.html", error=str(e))

@app.route("/feedback", methods=["POST"])
def feedback():
    request_id = request.form.get("request_id")
    feedback_score = request.form.get("feedback_score")

    try:
        response = requests.post(f"{BASE_URL}/feedback", json={"request_id": request_id, "feedback_score": feedback_score})
        response.raise_for_status()
        return jsonify({"message": "Feedback submitted successfully"})
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
