import pandas as pd
import torch
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = {
    "text": [
        "I love this product",
        "This movie is terrible",
        "The service was excellent",
        "I am very disappointed",

        "The food tastes amazing",
        "The experience was bad",
        "I really enjoyed the show",
        "This is the worst purchase",
        "Absolutely fantastic performance",
        "Not worth the money"
    ],
    "label": [1,0,1,0,1,0,1,0,1,0]
}
df = pd.DataFrame(data)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)
train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)
train_dataset.set_format(type="torch", columns=["input_ids","attention_mask","label"])
test_dataset.set_format(type="torch", columns=["input_ids","attention_mask","label"])
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_steps=10,
    save_strategy="epoch"
)
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    preds = torch.argmax(torch.tensor(logits), dim=1)
    return {"accuracy": accuracy_score(labels, preds)}
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)
print("Training model...")
trainer.train()
print("Evaluating model...")
results = trainer.evaluate()
print(results)
text = "This product is amazing"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
prediction = torch.argmax(outputs.logits).item()
label = "Positive" if prediction == 1 else "Negative"
print("Input:", text)
print("Prediction:", label)
