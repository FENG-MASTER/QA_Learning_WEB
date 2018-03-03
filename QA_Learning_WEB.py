from flask import Flask
from flask import render_template

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('Search.html')

@app.route('/result/')
def result():
    return render_template('Result.html')


if __name__ == '__main__':
    app.run()
