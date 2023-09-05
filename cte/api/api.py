from fastapi import FastAPI
from cte.ml_logic.load_model_and_generate import predict_message
#import model
app = FastAPI()

@app.get("/")
def home():
    return {"test" : True}

@app.get("/predict")
def predict(git_diff):
    prediction = predict_message(git_diff)
    return {"prediction" : prediction}
