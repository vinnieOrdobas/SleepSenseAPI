from flask import Flask, render_template, request
from models.predictors import Predictors

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form', methods=['GET'])
def form():
    return render_template('form.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    
    predictor = Predictors(form_data=form_data)

    inputs = predictor.parameter_handler.process_inputs()

    try:
        predictions = predictor.predict(inputs)
        response = predictor.parameter_handler.format_output(predictions)
    except Exception as error:
        response = {
            'error': str(error)
        }
    
    return render_template('result.html', prediction=response['prediction'], confidence=response['confidence'])

if __name__ == '__main__':
    app.run(debug=True)