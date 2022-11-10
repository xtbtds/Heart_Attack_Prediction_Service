import pickle
from fastapi import FastAPI
from pydantic import BaseModel

class Request(BaseModel):
    age: float
    sex: float
    cp: float
    trtbps: float
    chol: float
    fbs: float
    restecg: float
    thalachh: float
    exng: float
    oldpeak: float
    slp: float
    caa: float
    thall: float

model_file = 'model.bin'
with open (model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = FastAPI()
@app.post('/predict')
def predict(params:Request):
    analysis = params.dict()
    X = dv.transform(analysis)
    y_pred = model.predict_proba(X)[0,1]
    attack = y_pred >= 0.5
    result = {
        'attack_probability': float(y_pred),
        'attack': bool(attack)
    }
    return result


