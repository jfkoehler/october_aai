from joblib import load

model = load('logreg.joblib')

from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return 'Welcome home!'

@app.route('/linear_regression')
def lr():
    return f'''
    <h4>The coefficienct for the the model are {str(model.coef_[0][4])}</h4>
    '''

def contact():
    return 'Contact Me.'


if __name__ == '__main__':
    app.run(debug = True)