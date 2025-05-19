from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.preprocessing import preprocess
import joblib
import pandas as pd
import os

app = FastAPI()

# Mount UI folder
app.mount("/static", StaticFiles(directory="app/ui"), name="static")
templates = Jinja2Templates(directory="app/ui")

# Load model
model_path = None
for file in os.listdir("app"):
    if file.startswith("best_model") and file.endswith(".pkl"):
        model_path = os.path.join("app", file)
        break

model = joblib.load(model_path)

@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def get_prediction(request: Request,
                   gender: str = Form(...),
                   SeniorCitizen: int = Form(...),
                   Partner: str = Form(...),
                   Dependents: str = Form(...),
                   tenure: int = Form(...),
                   PhoneService: str = Form(...),
                   MultipleLines: str = Form(...),
                   InternetService: str = Form(...),
                   OnlineSecurity: str = Form(...),
                   OnlineBackup: str = Form(...),
                   DeviceProtection: str = Form(...),
                   TechSupport: str = Form(...),
                   StreamingTV: str = Form(...),
                   StreamingMovies: str = Form(...),
                   Contract: str = Form(...),
                   PaperlessBilling: str = Form(...),
                   PaymentMethod: str = Form(...),
                   MonthlyCharges: float = Form(...),
                   TotalCharges: float = Form(...)):

    data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    df = pd.DataFrame([data])
    processed = preprocess(df)
    print(processed)

    proba = model.predict_proba(processed)[0][1]
    print(f"Predicted probability of churn: {proba}")



    if proba >= 0.6:
        result = "Yes"
    elif proba < 0.6:
        result = "No"



    
    return templates.TemplateResponse("index.html", {"request": request, "result": result})
