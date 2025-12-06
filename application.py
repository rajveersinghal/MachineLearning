from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.pipeline.train_pipeline import train_pipeline


application = Flask(__name__)
app=application
# Home Route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
    gender=request.form.get('gender'),
    race_ethnicity=request.form.get('race_ethnicity'),
    parental_level_of_education=request.form.get('parental_level_of_education'),
    lunch=request.form.get('lunch'),
    test_preparation_course=request.form.get('test_preparation_course'),
    reading_score=float(request.form.get('reading_score')),
    writing_score=float(request.form.get('writing_score'))
)


        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)

        return render_template('home.html', results=result[0])


@app.route('/train', methods=['GET'])
def train_model():
    try:
        best_model_name, best_score, best_params = train_pipeline()

        return render_template(
            'train_result.html',
            model=best_model_name,
            score=best_score,
            params=best_params
        )
    except Exception as e:
        return f"Training failed: {str(e)}"


if __name__ == "__main__":
    app.run(host="0.0.0.0")
