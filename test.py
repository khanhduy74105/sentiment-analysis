# predict_sentiment.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./sentiment-model")
tokenizer = AutoTokenizer.from_pretrained("./sentiment-model")

# Prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    label = model.config.id2label[predicted_class_id]
    return label

# Example usage
if __name__ == "__main__":
    example_text = "this product ok, but not too much!"
    sentiment = predict_sentiment(example_text)
    print("Text:", example_text)
    print("Predicted Sentiment:", sentiment)
