from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, pipeline
# T5Tokenizer
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
import evaluate
from tokenizer import new_tokens_list, updated_tokenizer
# Model types and sizes
model_checkpoints = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b", "google/flan-t5-small"]
MODEL_CHECKPOINT = model_checkpoints[0]
model_type = AutoModelForSeq2SeqLM
# Data location, feature and target names
feature = "diff"
target = "message"
# location of data, please replace it with yuor own one
data_source = "/Users/omarkarame/code/OmarKarame/Commit-To-Excellence/Commit-To-Excellence-Backend/raw_data/test_output.json"
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
model_dir = f"../../saved_models/{model_name}"
# Testing
GIT_DIFF = "random diff that is going to be used to generate comment when committed"
# Read data
def get_data(url):
    df = pd.read_json(url)
    df = df.drop(columns=["sha"]).dropna()
    # df = df[df[feature].str.len() < 5000]
    df = df.iloc[0: 1000]
    data = Dataset.from_pandas(df, preserve_index=False)
    return data
# Create dataset objects
def split_data(data, validation_size, test_size):
    # Split data into train, validation and testing
    data_train_test = data.train_test_split(test_size=int(len(data)*test_size))
    data_train_val = data_train_test["train"].train_test_split(test_size=int(len(data)*validation_size))
    ds = DatasetDict({
        "train": data_train_val["train"],
        "validation": data_train_val["test"],
        "test": data_train_test["test"]})
    return ds
# Instantiate Tokenizer
def create_basic_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    return tokenizer
def preprocess_data(examples):
    # Process + tokenize features
    inputs = [prefix + doc for doc in examples[feature]]
    model_inputs = tokenizer(inputs, max_length=max_feature_length, truncation=True)
    print(type(model_inputs))
    # tokenize targets
    labels = tokenizer(examples[target], max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
def create_model_arguments(batch_size, weight_decay, num_of_epochs):
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
    return args
#Create data collator
def create_data_collator():
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
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
    return {k: round(v, 4) for k, v in result.items()}
# create T5 model
def create_t5_model(tokenizer):
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
    model.resize_token_embeddings(len(tokenizer))
    return model
# Create trainer model
def create_trainer(model, args, tokenized_datasets, data_collator, tokenizer):
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics)
    return trainer
# Save model
def save_trained_model():
    trainer.save_model("saved_models/t5-small-cte")
# Load trained model
def load_trained_model():
    loaded_model = AutoModelForSeq2SeqLM.from_pretrained("saved_models/t5-small-cte")
    return loaded_model
# Generate commit message
def generate_commit_message(diff, model, tokenizer):
    input = ["summarize: " + diff]
    #pred_text = "@staticmethod\n def __check_dictionary(word):\n '''Check if word exists in English dictionary'''\n response = requests.get(f'https://wagon-dictionary.herokuapp.com/{word}')\n json_response = response.json()\n        return json_response[\"found\"]"
    inputs = tokenizer(input, max_length=max_feature_length, truncation=True, return_tensors="pt")
    output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=64)
    decoder_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return decoder_output
if __name__ == '__main__':
    data = get_data(data_source)
    print("Data loaded")
    data_set = split_data(data, 0.2, 0.2)
    print("data split into train,validation and test")
    #tokenizer = create_basic_tokenizer()
    tokenizer = updated_tokenizer(new_tokens_list, MODEL_CHECKPOINT)
    print("tokenizer created")
    tokenized_datasets = data_set.map(preprocess_data, batched=True)
    print("Data tokenized")
    args = create_model_arguments(batch_size, weight_decay, num_of_epochs)
    print("arguments for training created")
    data_collator = create_data_collator()
    print("data collator created")
    model = create_t5_model(tokenizer)
    print("T5 model created")
    trainer = create_trainer(model, args, tokenized_datasets, data_collator, tokenizer)
    print("training model created")
    #Train model
    trainer.train()
    print("model trained")
    save_trained_model()
    print("model saved at saved_models/t5-small-cte")
    trained_model = load_trained_model()
    print("model loaded from saved_models/t5-small-cte")
    #prediction_text = "Tap into this collection of KS2 writing examples to support your teaching on writing all types of texts. Writing examplars are model texts of what KS2 children should be aiming to achieve. All of our KS2 writing exemplars include a breakdown of what's included, how text should be formatted and why it's important. In each resource, we've also added a detailed PowerPoint on the topic, word mats, exemplification checklist, genre features checklist and of course a brilliant example of a specific type of writing. Model texts are great resources for helping children to understand how their work is marked, and what they should strive towards completing. This helps towards providing writing inspiration and confidence, alongside aiding children to proof-read their own work and select areas for improvement. You can choose to go through these KS2 writing exemplars together or individually. All of our model texts provide detailed notes to follow to help guide children with their own work. You could also provide an exemplar to your children while they're undergoing their own writing activity to help guide their work to show your class the correct formatting. This will help your KS2 children to memorise a writing structure for their work. Dyslexia-friendly options are also included within this collection of resources. "
    comment = generate_commit_message(data_set["test"]["diff"][0], trained_model, tokenizer)
    model1 = create_t5_model(tokenizer)
    basic_comment = generate_commit_message(data_set["test"]["diff"][0], model1, tokenizer)
    print("predicted comment")
    print(comment)
    print(f"\n")
    print("predicted basic comment")
    print(basic_comment)
