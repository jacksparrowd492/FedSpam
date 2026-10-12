"""
FedSpam-Shield — Flask web interface for the federated global model.

Usage:
    python flask_app.py        # then open http://127.0.0.1:5000

Reuses predict.py for the model and feature pipeline; nothing is duplicated here.
"""

from flask import Flask, render_template, request

from predict import load_artifacts, predict

app = Flask(__name__)

# Load the model, TF-IDF vectorizer and scaler once at startup instead of per request.
ARTIFACTS = load_artifacts()


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    subject = ""
    body = ""
    probability = None

    if request.method == "POST":
        subject = request.form.get("subject", "").strip()
        body = request.form.get("body", "").strip()

        if not subject and not body:
            error = "Enter an email subject or body before checking."
        else:
            try:
                # Reuse the existing trained-model prediction function.
                result, probability = predict(subject, body, artifacts=ARTIFACTS)
                # Convert probability to a percentage for display.
                probability *= 100
            except Exception as exc:
                app.logger.exception("Prediction failed")
                error = f"Prediction failed: {exc}"
                result = None

    return render_template(
        "index.html",
        result=result,
        error=error,
        subject=subject,
        body=body,
        probability=probability,
    )


if __name__ == "__main__":
    # debug=False: the Werkzeug debugger allows code execution if exposed.
    app.run(host="127.0.0.1", port=5000, debug=False)
