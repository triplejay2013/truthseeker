import torch
import re
from transformers import DistilBertTokenizer, DistilBertModel, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score


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


# Load DistilBERT model
model = DistilBertClassifier()
if torch.cuda.is_available():
    model.cuda()

def train(train_dataset, test_dataset, valid_dataset, out: str):
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f'./results/{out}',
        fp16=True,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'./logs/{out}',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
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