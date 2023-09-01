from transformers import AutoTokenizer
from transformer import MODEL_CHECKPOINT

new_tokens_list = ["{", "}", "{ }"]

def updated_tokenizer(list_of_new_tokens):
    """Instantiate new tokenizer updated with data fed"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    tokenizer.add_tokens(list_of_new_tokens)

    return tokenizer
