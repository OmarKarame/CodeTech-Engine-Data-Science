from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, pipeline
# T5Tokenizer
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
import evaluate
from tokenizer import new_tokens_list, updated_tokenizer, special_tokens_dict



# Model types and sizes
model_checkpoints = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b", "google/flan-t5-small", "google/flan-t5-large"]
MODEL_CHECKPOINT = model_checkpoints[0]
model_type = AutoModelForSeq2SeqLM

# Data location, feature and target names
feature = "diff"
target = "message"

# location of data, please replace it with yuor own one
data_source = "test_data/test_output.json"
#"/home/tomasz/code/OmarKarame/Commit-To-Excellence-Backend/cte/tomasz_test/data2.csv"


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
model_dir = f"saved_models/{model_name}"



# Read data
def get_data(url):
    """Load data and create data set from selected json file"""

    df = pd.read_json(url)
    df = df.drop(columns=["sha"]).dropna()
    # df = df[df[feature].str.len() < 5000]
    df = df.iloc[0: 1000]
    data = Dataset.from_pandas(df, preserve_index=False)
    print("\u2713 data uploaded")
    return data


# Split data into train, validation and testing
def split_data(data, validation_size, test_size):
    """Split data into train, validation and testing. Returns dataset"""

    data_train_test = data.train_test_split(test_size=int(len(data)*test_size))
    data_train_val = data_train_test["train"].train_test_split(test_size=int(len(data)*validation_size))


    ds = DatasetDict({
        "train": data_train_val["train"],
        "validation": data_train_val["test"],
        "test": data_train_test["test"]})
    print("\u2713 data divided into train, validation and test")
    return ds


# Instantiate Tokenizer
def create_basic_tokenizer():
    """Instantiate tokenizer"""

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    print("\u2713 created basic tokenizer")
    return tokenizer


def preprocess_data(examples):
    """Tokenizes features and target"""

    inputs = [prefix + doc for doc in examples[feature]]
    model_inputs = tokenizer(inputs, max_length=max_feature_length, truncation=True)
    labels = tokenizer(examples[target], max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    print("\u2713 Data tokenized")
    return model_inputs


def create_model_arguments(batch_size, weight_decay, num_of_epochs):
    """Creates arguments needed to train the model"""

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
    print("\u2713 arguments for training created")
    return args


#Create data collator
def create_data_collator():
    """Creates data collator needed to train the model"""
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    print("\u2713 data collator created")
    return data_collator


# Load metric with evaluation function to determine the model performance
def compute_metrics(eval_pred):
    """Takes a tuple of predictions and reference labels as input,
    and outputs a dictionary of metrics computed over the inputs."""
    rouge = evaluate.load("rouge")
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    print("\u2713 metric created")
    return {k: round(v, 4) for k, v in result.items()}


# create T5 model
def create_t5_model(tokenizer):
    """Instantiates T5 model"""
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
    model.resize_token_embeddings(len(tokenizer))
    #pad_to_multiple_of=int((len(tokenizer)/16+1))*16
    print("\u2713 T5 model created")
    return model


# Create trainer model
def create_trainer(model, args, tokenized_datasets, data_collator, tokenizer):
    """Creates trainer by combinining model, arguments, tokenizer, data and data collator"""
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics)
    print("\u2713 training model created")
    return trainer


# Save model
def save_trained_model():
    path = "saved_models/t5-small"
    trainer.save_model(path)
    print(f"\u2713 model saved at {path}")


# Load trained model
def load_trained_model():
    path = "saved_models/t5-small"
    loaded_model = AutoModelForSeq2SeqLM.from_pretrained(path)
    # ignore_mismatched_sizes=True
    print(f"\u2713 model loaded from {path}")
    return loaded_model


# Generate commit message
def generate_commit_message(diff, model, tokenizer):
    input = ["summarize: " + diff]
    inputs = tokenizer(input, max_length=max_feature_length, truncation=True, return_tensors="pt")
    output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=64)
    decoder_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return decoder_output


if __name__ == '__main__':
    #Runt the model file from top to bottom, uploads data, split it,
    # create training model, train it, save it, load it and predict commit meesage
    data = get_data(data_source)

    data_set = split_data(data, 0.2, 0.2)
    #tokenizer = create_basic_tokenizer()
    tokenizer = updated_tokenizer(new_tokens_list, special_tokens_dict, MODEL_CHECKPOINT)

    tokenized_datasets = data_set.map(preprocess_data, batched=True)

    args = create_model_arguments(batch_size, weight_decay, num_of_epochs)

    data_collator = create_data_collator()

    model = create_t5_model(tokenizer)

    trainer = create_trainer(model, args, tokenized_datasets, data_collator, tokenizer)

    #Trains the model
    trainer.train()
    print("\u2713 model trained")

    save_trained_model()

    trained_model = load_trained_model()

    comment = generate_commit_message(data_set["test"]["diff"][0], trained_model, tokenizer)

    model1 = create_t5_model(tokenizer)

    basic_comment = generate_commit_message(data_set["test"]["diff"][0], model1, tokenizer)
    print("predicted comment")
    print(comment)
    print(f"\n")
    print("predicted basic comment")
    print(basic_comment)
