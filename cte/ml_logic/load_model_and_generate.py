from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformer import  generate_commit_message, data_source, get_data, create_t5_model
import pandas as pd

# TOKENIZER IS NOT WORKING CORRECTLY!
#path to model will need to be changed as is currently being saved outside of the project files(still uploaded on the main branch for now)

def load_data(data_source):
    df = pd.read_json(data_source)
    df = df.drop(columns=["sha"]).dropna()
    print(f'loaded frame of shape {df.shape}')

    return df
#def generate_commit_message():
choice = int(input("Pick data point:"))
model = AutoModelForSeq2SeqLM.from_pretrained("saved_models/t5-small")
tokenizer1 = AutoTokenizer.from_pretrained("saved_models/t5-small")
model.resize_token_embeddings(len(tokenizer1))


print("model loaded")
basic_model = create_t5_model(tokenizer1)
print("basic model loaded")

data = load_data(data_source)
print("data loaded)")

comment = generate_commit_message(data["diff"][choice], model, tokenizer1)
    #print(f"type of the comment: {type(comment)}")
print("Commit message predicted by trained T5 model:")
print(comment)
print(f"\n")



basic_comment = generate_commit_message(data["diff"][choice], basic_model, tokenizer1)
print("Commit message predicted by untrained model:")
print(basic_comment)
print(f"\n")

print("Actual commit message:")
print(data["message"][choice])
print(f"\n")

#if __name__ == '__main__':
    #generate_commit_message()
