# SleepSenseAPI
## [LIVE DEMO](https://sleepsenseapi.onrender.com/)

An app that detects whether an user has a sleep disorder, based on anonymised form data.

## Dependencies:


### Python Libraries
- flask
- gunicorn
- joblib
- pytest
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

## Installation:
1. **Clone the repository:**
```
git clone <repo_url>
cd <repo_directory>
```
2. **Download and Install Python:**
- Go to [Python downloads page](https://www.python.org/downloads/)
- Download the latest version for your machine (Windows, macOS, Linux)
- Run the installer and follow instructions
- Alternatively, use [brew](https://docs.brew.sh/):
```
brew install python3
```
3. **Create and activate environment:**
- Open terminal and navigate to project directory. Create and activate virtual environment, as such:
```
python3 -m venv <env_name>
source <env_name>/bin/activate
```
4. **Install requirements:**
- Install required libraries running:
```
pip install -r requirements.txt
```
5. **Run Flask server:**

- Uncomment line 38 in the `app.py` file:
```
# The following code is commented out because Render runs the application using gunicorn in production. If running locally, uncomment the code below:

# if __name__ == '__main__':
#     app.run(debug=True)
```

- Start Flask's development server:
```
cd API
python3 app.py
```
Server should be running on `http://127.0.0.1:5000/`