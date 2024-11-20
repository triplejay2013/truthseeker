import torch
import re
import os
import csv
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel, Trainer, TrainingArguments, TrainerCallback
from sklearn.metrics import accuracy_score
import nltk
nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer


def clean_text(text):
    text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    text = re.sub(r"[^a-zA-z.!?'0-9]", ' ', text)
    text = re.sub('\t', ' ',  text)
    text = re.sub(r" +", ' ', text)
    return text


def compute_consensus(row, majority_answer_category, target=4):
    """
    Compute consensus or truthfulness based on the target number of classes.
    
    Parameters:
        row (pd.Series): A row of the DataFrame.
        majority_answer_category (str): Column name for the majority answer
        target (int): Number of classification labels (2 or 4).
                      2 for binary truthfulness,
                      4 for detailed consensus.
    
    Returns:
        str: The computed label.
    """
    if target not in {2, 4}:
        raise ValueError("Invalid target value. Must be 2 or 4.")
    
    # Mapping based on comparison to Ground "Truth" (ie value of 1)
    true_mapping = {
        4: {
            "Agree": "True",
            "Mostly Agree": "Mostly True",
            "Disagree": "False",
            "Mostly Disagree": "Mostly False"
        },
        2: {
            "Agree": "True",
            "Mostly Agree": "True",
            "Disagree": "False",
            "Mostly Disagree": "False"
        }
    }
    
    false_mapping = {
        4: {
            "Agree": "False",
            "Mostly Agree": "Mostly False",
            "Disagree": "True",
            "Mostly Disagree": "Mostly True"
        },
        2: {
            "Agree": "False",
            "Mostly Agree": "False",
            "Disagree": "True",
            "Mostly Disagree": "True"
        }
    }
    
    mapping = true_mapping if row["BinaryNumTarget"] == 1 else false_mapping
    return mapping[target].get(row[majority_answer_category], None)

# Load DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize the tweet text
def tokenize_function(examples):
    return tokenizer(examples['full'], padding="max_length", truncation=True)

# Define compute_metrics function to calculate accuracy
def compute_metrics(p):
    preds = p.predictions.argmax(axis=1)  # Get predicted labels
    labels = p.label_ids  # Get true labels
    acc = accuracy_score(labels, preds)  # Compute accuracy
    return {"accuracy": acc}

# Define a custom model class that adds a classification head for binary classification
class DistilBertClassifier(torch.nn.Module):
    def __init__(self):
        super(DistilBertClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased", max_length=410, num_labels=2)
        self.classifier = torch.nn.Linear(self.distilbert.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask, labels=None):
        # Get hidden states from DistilBERT
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state
        # Use the [CLS] token's embedding for classification
        pooled_output = hidden_state[:, 0]  # First token is the [CLS] token
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            # Binary crossentropy loss
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.view(-1).float())  # Ensure labels are float for BCE loss
            return loss, logits
        else:
            return logits

# Custom callback to log metrics to CSV
class LogMetricsCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file
        # Ensure the directory exists and the file exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        # Check if the log file already exists; if not, write headers
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write("step,epoch,train_loss,eval_loss,train_accuracy,eval_accuracy\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Log metrics at the end of each step
        if logs is not None:
            logs_df = pd.DataFrame([{
                'step': state.global_step,
                'epoch': state.epoch,
                'train_loss': logs.get('loss', None),
                'eval_loss': logs.get('eval_loss', None),
                'train_accuracy': logs.get('train_accuracy', None),
                'eval_accuracy': logs.get('eval_accuracy', None)
            }])
            logs_df.to_csv(self.log_file, mode='a', header=False, index=False)


# Load DistilBERT model
model = DistilBertClassifier()
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    model.to(device)

def train(train_dataset, test_dataset, valid_dataset, out: str):
    log_file = f'./logs/{out}/training_metrics.csv'
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f'./results/{out}',
        fp16=True,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=1e-8,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
    )

    callbacks = [LogMetricsCallback(log_file)]
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # Train the model
    trainer.train()

    # Evaluate on the test dataset
    test_results = trainer.evaluate()
    print("Test Accuracy:", test_results['eval_accuracy'])

    # Get predictions on the validation dataset
    predictions = trainer.predict(valid_dataset)
    valid_accuracy = accuracy_score(valid_dataset['labels'], predictions.predictions.argmax(axis=-1))
    print("Validation Accuracy:", valid_accuracy)
    return test_results.get('eval_accuracy'), valid_accuracy

sia = SentimentIntensityAnalyzer()
_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def get_statement_embeddings(statements):
    embeddings = []
    for statement in statements:
        inputs = tokenizer(statement, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = _model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).numpy()
        embeddings.append(embedding)
    return np.concatenate(embeddings, axis=0)

def extract_features(tweets):
    sentiments = []
    tweet_embeddings = []

    for tweet in tweets:
        # Sentiment analysis
        sentiment_score = sia.polarity_scores(tweet)['compound']
        sentiments.append(sentiment_score)

        # Tokenize the tweet and get embeddings
        inputs = tokenizer(tweet, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = _model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).numpy()  # Mean pooling
        tweet_embeddings.append(embedding)

    tweet_embeddings = np.concatenate(tweet_embeddings, axis=0)
    avg_sentiment = np.mean(sentiments)
    avg_embedding = np.mean(np.vstack(tweet_embeddings), axis=0)

    return avg_sentiment, avg_embedding