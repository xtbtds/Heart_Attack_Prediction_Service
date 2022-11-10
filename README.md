# Heart Attack Prediction Service
*Dataset: https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset*  
### FastAPI app predicting the probability of heart attack based on these parameters: 
  
***Age*** - Age of the patient  
***sex*** - Sex of the patient  
***cp*** - Chest pain type ~ 0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal Pain, 3 = Asymptomatic  
***trtbps*** - Resting blood pressure (in mm Hg)  
***chol*** - Cholestoral in mg/dl fetched via BMI sensor  
***fbs*** - (fasting blood sugar > 120 mg/dl) ~ 1 = True, 0 = False  
***restecg*** - Resting electrocardiographic results ~ 0 = Normal, 1 = ST-T wave normality, 2 = Left ventricular hypertrophy  
***thalachh*** - Maximum heart rate achieved  
***oldpeak*** - Previous peak  
***slp*** - Slope  
***caa*** - Number of major vessels  
***thall*** - Thalium Stress Test result ~ (0,3)  
***exng*** - Exercise induced angina ~ 1 = Yes, 0 = No  
***output*** - Target variable  
    
### To run the app using docker compose:  
```
docker-compose up -d
```
### Then run jupyter notebook named `test_service.ipynb`

