# Heart Attack Prediction Service
*Dataset: https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset*  

- FastAPI app predicting the probability of heart attack. Trained 2 models: Random Forest and Logistic Regression with parameter tuning and cross validation. The best *AUC-ROC* score is ***87%***. NOTE, that due to the lack of data, model works correctly only with specific range of each parameter (f.e, there were no rows for age <= 30 in train dataset, so I do not recommend to use it seriously). Run `notebook.ipynb` to see what does the data look like.

- Parameters:

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


- To run the app using docker compose:  
```
- cd <project_folder>
- docker-compose up -d
```

- To run the project using docker:
```
- docker build -t <image_name_tag>
- docker images | grep <image_name_tag>   #check built image
- docker run -p 9696:9696 -it --rm  <image_name_tag>
```

- Then run jupyter notebook named `test_service.ipynb` to predict result on your data.

## Virtual environment pipenv (optional)
If you want to run the app in virtual environment, uncomment these 2 lines in `Dockerfile`:
```
RUN pip install pipenv
RUN pipenv install --deploy --system
```
And make sure that you removed this line:
```
# RUN pip install -r requirements.txt
```
The repository already has `Pipfile` and `Pipfile.lock`.
