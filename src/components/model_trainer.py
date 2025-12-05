import os
import sys 

from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,
                              GradientBoostingRegressor,
                              RandomForestRegressor)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score

from src.exception  import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models



@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]

            models = {
                        "Random Forest": RandomForestRegressor(),
                        "Decision Tree": DecisionTreeRegressor(),
                        "Gradient Boosting": GradientBoostingRegressor(),
                        "Logistic Regression": LogisticRegression(),
                        "KNN Regressor": KNeighborsRegressor(),
                        "XGBoost Regressor": XGBRegressor(objective='reg:squarederror'),
                        "CatBoost Regressor": CatBoostRegressor(verbose=False),
                        "AdaBoost Regressor": AdaBoostRegressor(),
                    }
            
            params = {
                "Random Forest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [6, 10, None],
                    "min_samples_split": [2, 5, 10],
                },

                "Decision Tree": {
                    "max_depth": [4, 6, 8, 10],
                    "criterion": ["squared_error", "friedman_mse"],
                },

                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5],
                },

                "Logistic Regression": {
                    "max_iter": [100, 200, 300],
                    "solver": ["lbfgs"],
                },

                "KNN Regressor": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                },

                "XGBoost Regressor": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.03, 0.05, 0.1],
                    "max_depth": [3, 6, 10],
                    "subsample": [0.7, 0.8],
                },

                "CatBoost Regressor": {
                    "depth": [4, 6, 8],
                    "learning_rate": [0.03, 0.05],
                    "iterations": [200, 300],
                },

                "AdaBoost Regressor": {
                    "n_estimators": [50, 100, 150],
                    "learning_rate": [0.03, 0.05, 0.1],
                },
            }



            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models,param=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found: {best_model_name} with score {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Model saved successfully")

            predicted = best_model.predict(X_test)
            r2_score_score = r2_score(y_test, predicted)
            return r2_score_score,best_model_name
        
        except Exception as e:
            raise CustomException(e, sys)



