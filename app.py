from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form', methods=['GET'])
def form():
    return render_template('form.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = request.form.get('data')
    # Here, you can process the data, e.g., pass it to your model for prediction
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)