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
- docker build -t <image_name>
- docker images | grep <image_name>   #check built image
- docker run -p 9696:9696 -it --rm  attack:v1
```
- If you have en error `docker: Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?` try this:
```
- sudo dockerd
```
- Then run jupyter notebook named `test_service.ipynb` to predict result on your data.

