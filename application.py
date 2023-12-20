from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import PredictPipeline
import pandas as pd
application = Flask(__name__)
app = application

pipeline = PredictPipeline()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    team = int(request.form['team'])
    position = int(request.form['position'])
    age = int(request.form['age'])
    appearance = float(request.form['appearance'])
    minutes_played = float(request.form['minutes_played'])
    games_injured = int(request.form['games_injured'])
    award = int(request.form['award'])
    current_value = int(request.form['current_value'])
    
    user_data = pd.DataFrame({
        'team': [team],
        'position': [position],
        'age': [age],
        'appearance': [appearance],
        'minutes played': [minutes_played],
        'games_injured': [games_injured],
        'award': [award],
        'current_value': [current_value],
    })

    # Preprocess user input
    user_data = pipeline.preprocess_input(user_data)

    # Make a prediction
    prediction = pipeline.make_prediction(user_data)
    
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host="0.0.0.0")