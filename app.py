from flask import  Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained machine learning model
with open('Lab4.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        weight = request.form['weight']
        length1 = request.form['length1']
        length2 = request.form['length2']
        length3 = request.form['length3']
        height = request.form['height']
        width = request.form['width']

        values = [weight, length1, length2, length3, height, width]
        values = np.array(values).reshape(-1, 6)
        prediction = model.predict(values)
        result = str
        for item in prediction:
            result = item

        return redirect(url_for('result', predict=result))
    else:
        return render_template('index.html')

@app.route('/result/<predict>')
def result(predict):
    return render_template('answer.html', answer=predict)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
