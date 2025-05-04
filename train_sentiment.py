import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)

# 1. Load dữ liệu
df = pd.read_csv("reviews-cleaned.csv")  # Cột: text, label

label2id = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
id2label = {v: k for k, v in label2id.items()}

df['Label'] = df['Label'].map(label2id)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=42)

# 2. Convert sang HuggingFace Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 3. Load tokenizer
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    tokens = tokenizer(example['Description'], truncation=True, padding='max_length', max_length=128)
    tokens["labels"] = example["Label"]
    return tokens


train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# 4. Load model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=3,
    id2label={0: "Negative", 1: "Neutral", 2: "Positive"},
    label2id={"Negative": 0, "Neutral": 1, "Positive": 2})

# 5. Huấn luyện
training_args = TrainingArguments(
    output_dir="./sentiment-model",
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

# 6. Lưu model
model.save_pretrained("./sentiment-model")
tokenizer.save_pretrained("./sentiment-model")
