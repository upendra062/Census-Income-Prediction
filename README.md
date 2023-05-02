# End To End ML Project
```
Census-Income Prediction
```

## Creating Environment
```
conda create -p venv python=3.8 -y
```
## Activate Environment
```
conda activate venv\
```

# Deployment WebApplication
```
Azure Web App Deploy
```

# DATABASE
```
MongoDB
```    
# Project LifeCycle
# Process of Machine Learning Project
```
    [DATA CLEANING] 
        { 
            I/P :RAW DATA, 
            O/P : CLEAN DATA,
        }   
    [DATA INJESTION]
        {
            I/P: CLEAN DATA,
            O/P: TRAIN AND TEST DATA 
            TRAIN TEST SPLIT
            We get train and test data
        }
    [DATA TRANSFORMATION]
        {
            WHAT WE DO HERE
            WE DO FEATURE ENGINEERING
            I/P: TRAIN AND TEST DATA
            O/P: TRAIN AND TEST DATA
        }
    [MODEL TRAINING]
        {
            I/O: TRAIN AND TEST DATA
            O/P: PICKLE FILE
        }
    [MODEL EVALUATION]
        {
            I/O: PICKLE FILE
            O/P: ACCURACY
        }
    [DEPLOYMENT]
```
# Steps
```
1. EDA
2. Feature Engineering
3. Model Training
4. Model Evaluation
4. Web Application Deployment
```

# Components
```
1. Data Cleaning 
``` 
```
2. Data Ingestion 
``` 
```
3. Data Transformation 
```
```
4. Model Training 
```
```
5. Model Evaluation 
```
```
6. Deployment
```


# TRAINING PIPELINE

```
DATA CLEANING
```
```
DATA INJESTION
```
```
DATA TRANSFORMATION
```
```
CLUSTERING (grouping the data)
```
```
MODEL TRAINING FOR EACH GROUP DATA
```
```
MODEL EVALUTION
```
```
HYPERPARAMTER TUNNING
```
```
FINDING BEST MODEL
```
```
SAVING BEST MODEL
```


# PREDICTION PIPELINE
--------------------

```
DATA TRANSFORMATION
 (because for the feature engineering we will also do for new data)
```
```
FEATURE ENGINEERING
```
```
PREDICTION
    I/O: DATA
        DATA PASSING THROUGH THE MODEL
    O/P: PREDICTION
```


# Runn Docker
```
docker build -t incomeprediction:latest .
```
### To Run Docker Image
```
docker run -p 5000:5000 incomeprediction
```
```
docker run -p 8888:5000 timeprediction
```
### Port In which Docker Image is Running
```
http://127.0.0.1:5000/
```
```
http://127.0.0.1:8888/
```

# DeployMent Step


##  azure deployment through github

```
search for 
1. web app 
2. create
```
# Steps for azure

```
enable github
connect it
select the repository
python 3.8
create
```

# 2 Directory(.github,workflows) and 1 file(master_{appname}.yml) is created . 
```
.github(directory)
---workflows(directory)
------master_{appname}.yml(file)

```