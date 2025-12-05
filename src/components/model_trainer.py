import os
import sys 

from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (AdaBoostClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor

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
                "Random Forest": RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                ),

                "Decision Tree": DecisionTreeClassifier(
                    criterion="gini",
                    max_depth=8,
                    min_samples_split=4
                ),

                "Gradient Boosting": GradientBoostingClassifier(
                    learning_rate=0.05,
                    n_estimators=200,
                    max_depth=3
                ),

                "Logistic Regression": LogisticRegression(
                    max_iter=200,
                    solver="lbfgs"
                ),

                "KNN Classifier": KNeighborsClassifier(
                    n_neighbors=7,
                    metric="minkowski"
                ),

                "XGBoost Regressor": XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror'
                ),

                "CatBoost Classifier": CatBoostClassifier(
                    iterations=300,
                    learning_rate=0.05,
                    depth=6,
                    verbose=False
                ),

                "AdaBoost Classifier": AdaBoostClassifier(
                    n_estimators=150,
                    learning_rate=0.05
                ),
            }


            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

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



