from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from cte.ml_logic.transformer import  generate_commit_message, data_source, get_data, create_t5_model
import pandas as pd
import os


cwd = os.getcwd()

# TOKENIZER IS NOT WORKING CORRECTLY!
#path to model will need to be changed as is currently being saved outside of the project files(still uploaded on the main branch for now)

def load_data(data_source):
    df = pd.read_json(data_source)
    df = df.drop(columns=["sha"]).dropna()
    print(f'loaded frame of shape {df.shape}')

    return df

if __name__ == '__main__':
    choice = int(input("Pick data point:"))
    model = AutoModelForSeq2SeqLM.from_pretrained(cwd +"/saved_models/t5-small-cte")
    tokenizer1 = AutoTokenizer.from_pretrained(cwd + "/saved_models/t5-small-cte")
    model.resize_token_embeddings(len(tokenizer1))


    print("model loaded")
    basic_model = create_t5_model(tokenizer1)
    print("basic model loaded")

    data = load_data(data_source)
    print("data loaded)")

    comment = generate_commit_message(data["diff"][choice], model, tokenizer1)
    #print(f"type of the comment: {type(comment)}")
    print("predicted comment")
    print(comment)
    print(f"\n")



    basic_comment = generate_commit_message(data["diff"][choice], basic_model, tokenizer1)
    print("predicted basic comment")
    print(basic_comment)
    print(f"\n")

    print("actual comment")
    print(data["message"][choice])

def predict_message(diff):
    model = AutoModelForSeq2SeqLM.from_pretrained(cwd +"/saved_models/t5-small-cte")
    tokenizer1 = AutoTokenizer.from_pretrained(cwd + "/saved_models/t5-small-cte")
    model.resize_token_embeddings(len(tokenizer1))

    print("model loaded")
    basic_model = create_t5_model(tokenizer1)
    print("basic model loaded")

    data = load_data(data_source)
    print("data loaded)")

    comment = generate_commit_message(diff, model, tokenizer1)

    return comment
