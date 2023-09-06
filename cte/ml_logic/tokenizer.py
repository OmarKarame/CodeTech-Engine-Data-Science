from transformers import AutoTokenizer

new_tokens_list = ["{", "}", "{ }", ",", ".", "'", "\n", "\t", "as", "assert", "def", "for", "continue", "def", "del", "elif", "else", "False", "finally", "import", "lambda", "None", "nonlocal", "raise", "return", "True", "while", "yield"]
special_tokens_dict = {"additional_special_tokens" : ["[sad]", "[ead]", "[ssb]", "[esb]", "[scn]", "[ecn]"]}

# words already included in tokenizer: and, break, class, except, for, from, if, in, is, nor, or, pass, try, with
def updated_tokenizer(list_of_new_tokens, special_tokens_dict, model_type):
    """Instantiate new tokenizer updated with data fed"""
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    list_of_new_tokens = set(new_tokens_list) - set(tokenizer.vocab.keys())
    special_tokens_dict["additional_special_tokens"] = list(set(special_tokens_dict["additional_special_tokens"]) - set(tokenizer.vocab.keys()))
    tokenizer.add_tokens(list(list_of_new_tokens))
    tokenizer.add_special_tokens(special_tokens_dict)

    return tokenizer


#tokenizer = updated_tokenizer(new_tokens_list, special_tokens_dict, "t5-small")
#print(type(tokenizer))
