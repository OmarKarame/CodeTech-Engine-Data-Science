from transformers import AutoTokenizer
from transformer import MODEL_CHECKPOINT

new_tokens_list = []

def updated_tokenizer(*args):
    """Instantiate new tokenizer updated with data fed"""
    basic_tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    new_tokenizer = basic_tokenizer.add_tokens(args)

    return new_tokenizer

updated_tokenizer(new_tokens_list)
