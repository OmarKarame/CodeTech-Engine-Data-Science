from fastapi import FastAPI
#import model
app = FastAPI()

@app.get("/")
def home():
    return {"test" : True}

@app.get("/predict")
def predict(git_diff):
    return {"prediction" : "temp"}
