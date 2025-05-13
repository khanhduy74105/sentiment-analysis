from flask import Flask, request, jsonify
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from flask_cors import CORS
# Load model và tokenizer từ local
MODEL_PATH = "my-finetuned-model"  # Đổi thành tên thư mục model bạn đã lưu

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Khởi tạo pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Flask app
app = Flask(__name__)
CORS(app)
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"]
    try:
        result = classifier(text)[0]
        return jsonify({
            "text": text,
            "label": result["label"],
            "confidence": round(result["score"], 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
