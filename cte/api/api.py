from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
import uvicorn as uvicorn
from fastapi import FastAPI
# from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List
# import os
# import sys

from cte.predict.preprocess_predict import full_diff_preprocessor
from cte.ml_logic.transformer import generate_commit_message

app = FastAPI()

# class RequestItem(BaseModel):
#     instance : str

# class ResponseItem(BaseModel):
#     prediction : str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.state.model = AutoModelForSeq2SeqLM.from_pretrained("../saved_models/untrained_large_cte_model")
app.state.tokenizer = AutoTokenizer.from_pretrained("../saved_models/untrained_large_cte_model")

# def generate_commit_message(diff, model, tokenizer):
#     input = ["summarize: " + diff]
#     inputs = tokenizer(input, max_length=256, truncation=True, return_tensors="pt")
#     output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=64)
#     decoder_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
#     return decoder_output


example = """--git a/.github/workflows/main.yml b/.github/workflows/main.yml
index 40f27afe..546cb46e 100644
--- a/.github/workflows/main.yml
+++ b/.github/workflows/main.yml
@@ -27,5 +27,6 @@ jobs:
       with:
         target_branch: gh-pages
         build_dir: public
+        fqdn: python.swaroopch.com
       env:
         GITHUB_PAT: ${{ secrets.GITHUB_PAT }}"""

#print(generate_commit_message(example, model, tokenizer))



@app.get("/")
def home():
    return {"test" : True}

@app.get("/predict")
def predict(git_diff):
    model = app.state.model
    tokenizer = app.state.tokenizer

    cleaned_diff = full_diff_preprocessor(git_diff)
    prediction = generate_commit_message(cleaned_diff, model, tokenizer, max_feature_length=256)
    return {"prediction" : prediction}



# def custom_openapi():
#     if app.openapi_schema:
#         return app.openapi_schema
#     openapi_schema = get_openapi(title="FastAPI", version="0.1.0", routes=app.routes)
#     openapi_schema["x-google-backend"] = {
#         "address" :"${CLOUD_RUN_URL}",
#         "deadline" : "${TIMEOUT}",
#     }
#     openapi_schema["paths"]["/predictions"]["options"] = {
#         "operationID": "corsHelloWorld",
#         "responses" : {"description": "Successful response"}
#     }
#     app.openapi_schema = openapi_schema
#     return app.openapi_schema

#app.openapi = custom_openapi
