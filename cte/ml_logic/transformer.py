from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, pipeline
# T5Tokenizer
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
import evaluate



# Model types and sizes
model_checkpoints = ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]
MODEL_CHECKPOINT = model_checkpoints[0]
model_type = AutoModelForSeq2SeqLM

# Data location, feature and target names
feature = "Diff"
target = "Message"
data_source = "/cte/tomasz_test/data2.csv"

# Preprocess + Tokenizer Params
max_feature_length = 256
max_target_length = 128
prefix = "summarize: "

# Fine-Tuning Parameters
batch_size = 8
learning_rate = 4e-5
#adam_beta1 = 0.9
weight_decay = 0.01
num_of_epochs = 3

# Model Saving parameters
model_name = "cte_model"
model_dir = f"../../saved_models/{model_name}"

# Testing
GIT_DIFF = "random diff that is going to be used to generate comment when committed"


# Read data
df = pd.read_csv(data_source)
df = df.drop(columns=["Unnamed: 0", "Repository"]).dropna()
df.head()

df = df[df[feature].str.len() < 5000]

# set validation and test size
VALIDATION_DATA_SIZE = int(len(df)*0.2)
TEST_DATA_SIZE = int(len(df)*0.2)


# Create dataset objects
data = Dataset.from_pandas(df, preserve_index=False)

# Split data into train, validation and testing
data_train_test = data.train_test_split(test_size=50)
data_train_val = data_train_test["train"].train_test_split(test_size=55)


ds = DatasetDict({
    "train": data_train_val["train"],
    "validation": data_train_val["test"],
    "test": data_train_test["test"]
})

# Instantiate Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)




def preprocess_data(examples):
    # Process + tokenize features
    inputs = [prefix + doc for doc in examples[feature]]
    model_inputs = tokenizer(inputs, max_length=max_feature_length, truncation=True)

    print(type(model_inputs))

    # tokenize targets
    labels = tokenizer(examples[target], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

tokenized_datasets = ds.map(preprocess_data, batched=True)
tokenized_datasets





args = Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=200,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=weight_decay,
    save_total_limit=3,
    num_train_epochs=num_of_epochs,
    predict_with_generate=True,
    #adam_beta1 = adam_beta1
    # fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1"
)

#Create data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

# Load metric with evaluation function to determine the model performance

rouge = evaluate.load("rouge")


def compute_metrics(eval_pred):
    """Takes a tuple of predictions and reference labels as input,
    and outputs a dictionary of metrics computed over the inputs."""

    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}



# create T5 model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)


# Create trainer model
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics)

# Train model
trainer.train()

# Save model
trainer.save_model("../../saved_models/t5-small-cte")


# Load trained model

pipe = pipeline("summarization", model="../../saved_models/t5-small-cte")

pipe(
    "@staticmethod\n def __check_dictionary(word):\n '''Check if word exists in English dictionary'''\n response = requests.get(f'https://wagon-dictionary.herokuapp.com/{word}')\n json_response = response.json()\n        return json_response[\"found\"]", max_length=30
)


model1 = AutoModelForSeq2SeqLM.from_pretrained("../../saved_models/t5-small-cte")


pred_text = "@staticmethod\n def __check_dictionary(word):\n '''Check if word exists in English dictionary'''\n response = requests.get(f'https://wagon-dictionary.herokuapp.com/{word}')\n json_response = response.json()\n        return json_response[\"found\"]"

inputs = ["summarize: " + pred_text]

inputs = tokenizer(inputs, max_length=max_feature_length, truncation=True, return_tensors="pt")
output = model1.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=64)
decoder_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
decoder_output


# Base prediction on untrained T5 model
as_list = pd.read_csv("data2.csv").dropna()
as_list = as_list["Diff"].tolist()

import random

i = 0
num_indexes = 10
rand_indexes = []


while i < num_indexes:
    r = random.randint(0, len(as_list) - 1)

    if r not in rand_indexes:
        rand_indexes.append(r)
        i += 1

rand_indexes

t5_basic = pipeline("summarization", MODEL_CHECKPOINT)

for i, ind in enumerate(rand_indexes):
    print(i, t5_basic(as_list[ind], max_length=30)[0]["summary_text"])
