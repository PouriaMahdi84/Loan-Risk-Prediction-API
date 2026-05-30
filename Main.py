from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
import logging

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

model = joblib.load("loan_model.pkl")
logger.info("Model loaded successfully")

class LoanData(BaseModel):
    Gender: int
    Married: int
    Dependents: float
    Education: int
    Self_Employed: int
    ApplicantIncome: int
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: int

@app.get("/")
def root():
    logger.info("Health check endpoint called")
    return {"message": "Loan Risk Prediction API"}

@app.post("/predict")
def predict(data: LoanData):
    try:
        logger.info(f"Prediction request received: {data}")
        
        features = np.array([[
            data.Gender,
            data.Married,
            data.Dependents,
            data.Education,
            data.Self_Employed,
            data.ApplicantIncome,
            data.CoapplicantIncome,
            data.LoanAmount,
            data.Loan_Amount_Term,
            data.Credit_History,
            data.Property_Area
        ]])

        prediction = model.predict(features)[0]
        result = "Approved" if prediction == 1 else "Rejected"
        
        logger.info(f"Prediction result: {result}")
        
        return {"loan_approval_prediction": result}
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {"error": str(e)}