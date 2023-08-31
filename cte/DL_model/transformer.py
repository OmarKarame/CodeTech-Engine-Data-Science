from transformers import T5Tokenizer, TFT5ForConditionalGeneration, AutoModelForSeq2SeqLM, AutoModelForCausalLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer

# Model types and sizes
MODEL_SIZES = ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]
MODEL_SIZE = MODEL_SIZES[0]
MODEL_TYPES = [TFT5ForConditionalGeneration, AutoModelForSeq2SeqLM, AutoModelForCausalLM ]
MODEL_TYPE = MODEL_TYPES[0]

# Data
FEATURES = "Differences"
TARGET = "Comments"
DATA = "Data"

# Determine task for teh model (summarize, translate)
PREFIX = "summarize: "

# Determine maximum input for features and the target
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 128

# Fine-Tuning Parameters
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
ADAM_BETA1 = 0.9
WEIGHT_DECAY = 0.01

# Testing
GIT_DIFF = "random diff that is going to be used to generate comment when committed"




# create the model
model = MODEL_TYPE.from_pretrained(MODEL_SIZE)

# create the tokenizer
tokenizer = T5Tokenizer.from_pretrained(MODEL_SIZE)

# function tokenizing data
def preprocess_function(df):
    # Setup the tokenizer for features
    inputs = [PREFIX + difference for difference in df[FEATURES]]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=df[TARGET], max_length=MAX_TARGET_LENGTH, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# tokenized data
data = DATA.map(preprocess_function, batched=True)

#Function splitting data
def split_data(data):
    train_data = data.iloc[[0, int(len(data)*0.7)]]
    validation_data = data.iloc[[int(len(data)*0.7), int(len(data)*0.9)]]
    test_data = data.iloc[[int(len(data)*0.9), len(data)-1]]
    return train_data, validation_data, test_data

#Split data
train_data, validation_data, test_data = split_data(data)


# Create new training model based on T5 model and given new parameters
def create_training_model(model, tokenizer, train_data, validation_data):

    # create arguments for training the model
    args = Seq2SeqTrainingArguments(
        f"{MODEL_SIZE}-finetuned-xsum",
        evaluation_strategy = "epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size= BATCH_SIZE,
        per_device_eval_batch_size= BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        adam_beta1 = ADAM_BETA1,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False)

    # pad inputs and labels to the maximum length in the batch
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # model creation
    trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=train_data,
    eval_dataset=validation_data,
    data_collator=data_collator,
    tokenizer=tokenizer)
    # compute_metrics=compute_metrics not sure whether it is needed so not included

    # return new training model
    return trainer

#Commit to Excellence model
CTE_model = create_training_model(model, tokenizer, train_data, validation_data)

# train CTE_model
CTE_model.train()


#Testing

def generate_comment(tokenizer, git_diff):
    preprocessed_text= PREFIX + git_diff

    encoding_test = tokenizer.encode(preprocessed_text,return_tensors="tf")

    summary_ids = CTE_model.generate(encoding_test, min_length=60, max_length=80)

    commit_message = tokenizer.decode(summary_ids[0])

    return commit_message

generate_comment(tokenizer, GIT_DIFF)
