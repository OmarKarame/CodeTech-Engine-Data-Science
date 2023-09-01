from transformers import AutoTokenizer

new_tokens_list = ["{", "}", "{ }", "[sad]", "[ead]", "[ssb]", "[esb]", "[scn]", "[ecn]", r"\n", r"\t", "as", "assert", "def", "for", "continue", "def", "del", "elif", "else", "False", "finally", "import", "lambda", "None", "nonlocal", "raise", "return", "True", "while", "yield"]

# words already included in tokenizer: and, break, class, except, for, from, if, in, is, nor, or, pass, try, with
def updated_tokenizer(list_of_new_tokens, model_type):
    """Instantiate new tokenizer updated with data fed"""
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    list_of_new_tokens = set(new_tokens_list) - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(list_of_new_tokens))

    return tokenizer


