# run pip install llama-cpp-python
#need to be added to requirements


from llama_cpp import Llama
import pandas as pd




data = pd.read_json(data_source)

data_lst = data["diff"].tolist()
data_lst[100]

llm = Llama(
    model_path=model_path
    n_ctx=1026,)
