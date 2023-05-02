import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')
            
            preprocessor = load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 age:float,
                 fnlwgt:float,
                 education_num:float,
                 capital_gain:float,
                 capital_loss:float,
                 hours_per_week:float,
                 workclass:str,
                 marital_status:str, 
                 occupation:str,
                 relationship:str,
                 race:str,
                 sex:str,
                 native_country:str
                 ):
        self.age = age
        self.fnlwgt = fnlwgt
        self.education_num =  education_num
        self.capital_gain =  capital_gain
        self.capital_loss = capital_loss 
        self.hours_per_week = hours_per_week
        self.workclass =  workclass
        self.marital_status =  marital_status
        self.occupation =  occupation
        self.relationship = relationship
        self.race = race
        self.sex = sex
        self.native_country = native_country

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'age':[self.age],
                'fnlwgt':[self.fnlwgt], 
                'education_num' :[self.education_num],
                'capital_gain':[self.capital_gain],
                'capital_loss':[self.capital_loss],
                'hours_per_week':[self.hours_per_week],
                'workclass':[self.workclass],
                'marital_status':[self.marital_status],
                'occupation':[self.occupation],
                'relationship':[self.relationship],
                'race':[self.race],
                'sex':[self.sex],
                'native_country':[self.native_country],
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('DataFrame Gathered')
            return df

        except Exception as e:
            logging.info("Exception Occured in prediction pipeline")
            raise CustomException(e, sys)
        

