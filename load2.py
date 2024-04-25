import xml.etree.ElementTree as ET
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

#parses XML files (4 downloaded files from website: https://zenodo.org/records/1489920)
def parse_xml_articles(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    articles_data = []
    for article in root.findall('article'):
        article_data = {
            'id': article.get('id'),
            'text': ''.join(p.text for p in article.findall('.//p') if p.text)
        }
        articles_data.append(article_data)
    return pd.DataFrame(articles_data)

def parse_xml_labels(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    labels_data = []
    for article in root.findall('article'):
        label_data = {
            'id': article.get('id'),
            'hyperpartisan': article.get('hyperpartisan') == 'true',
            'bias': article.get('bias')
        }
        labels_data.append(label_data)
    return pd.DataFrame(labels_data)

# Replace 'path_to_your_files' with the actual paths to your XML files
# Here is the download link from the website: https://zenodo.org/records/1489920 
# Training dataset and training ground truth is a piar, validation dataset and validation ground truth is a pair (their IDs correspond)
train_articles = parse_xml_articles(r'C:\Users\pengu\OneDrive - Georgia Institute of Technology\Spring 2024\CS 4650\Project\CS-4650-Final\data\articles-training-bypublisher-20181122\articles-training-bypublisher-20181122.xml')
train_labels = parse_xml_labels(r'C:\Users\pengu\OneDrive - Georgia Institute of Technology\Spring 2024\CS 4650\Project\CS-4650-Final\data\ground-truth-training-bypublisher-20181122\ground-truth-training-bypublisher-20181122.xml')
validation_articles = parse_xml_articles(r'C:\Users\pengu\OneDrive - Georgia Institute of Technology\Spring 2024\CS 4650\Project\CS-4650-Final\data\articles-validation-bypublisher-20181122\articles-validation-bypublisher-20181122.xml')
validation_labels = parse_xml_labels(r'C:\Users\pengu\OneDrive - Georgia Institute of Technology\Spring 2024\CS 4650\Project\CS-4650-Final\data\ground-truth-validation-bypublisher-20181122\ground-truth-validation-bypublisher-20181122.xml')

# Merge articles and labels on ID
train_data = pd.merge(train_articles, train_labels, on='id')
validation_data = pd.merge(validation_articles, validation_labels, on='id')

# Convert to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_data)
validation_dataset = Dataset.from_pandas(validation_data)

# Tokenization
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_validation = validation_dataset.map(tokenize_function, batched=True)

# Model
# BERT model specialized for sequence classification tasks (pre-trained)
# The training process below fine-tunes this BERT model (fine tunes its parameteres) to fit "https://huggingface.co/datasets/hyperpartisan_news_detection"
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)


#for analysis later on
def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}




# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy='epoch'
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
evaluation_results = trainer.evaluate()

print(evaluation_results)

predictions_output = trainer.predict(tokenized_validation)
predictions = predictions_output.predictions.argmax(axis=-1)


# Getting the labels from the dataset
true_labels = np.array(tokenized_validation["labels"])

# Finding where predictions and true labels differ
mislabeled_indices = np.where(predictions != true_labels)[0]

# Analyze mislabeled examples
for index in mislabeled_indices:
    print(f"ID: {validation_data.iloc[index]['id']}")
    print(f"Text: {validation_data.iloc[index]['text']}")
    print(f"True label: {true_labels[index]}, Predicted label: {predictions[index]}")
    print("\n")


