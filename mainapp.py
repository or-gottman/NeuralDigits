import base64
import io
import pickle
import gzip
from flask import Flask
from flask import render_template
from flask import request
import numpy as np
import digitsneuralnetwork
from PIL import Image
import matplotlib.image as mpimg


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


rounds = 3  # neural network training rounds

data = "./mnist.pkl.gz"
f = gzip.open(data, 'rb')
tr_d, va_d, te_d = pickle.load(f, encoding='latin1')
f.close()

training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
training_results = [vectorized_result(y) for y in tr_d[1]]
training_data = list(zip(training_inputs, training_results))
validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
validation_data = list(zip(validation_inputs, va_d[1]))
test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
test_data = list(zip(test_inputs, te_d[1]))



# training_data, validation_data, test_data = load_data.load_data()
net = digitsneuralnetwork.DigitsNeuralNetwork([784, 30, 10])
net.gradient_descent(training_data, rounds, 10, 3.0, test_data)

app = Flask(__name__)


@app.route('/', methods = ['GET'])
def paint():
    return render_template("paint.html")

@app.route('/paint.css', methods = ['GET'])
def css():
    return render_template("paint.css")

@app.route('/paint.js', methods = ['GET'])
def js():
    return render_template("paint.js")


@app.route('/recognize', methods = ['POST'])
def recognize():

    buffer = io.BytesIO()
    imgdata = base64.b64decode(request.form['data'])
    img = Image.open(io.BytesIO(imgdata))
    new_img = img.resize((28, 28))  # x, y
    new_img.save(buffer, format="PNG")

    new_img.save("input.png")
    img = mpimg.imread("input.png")

    newimg = []
    for row in img:
        for col in row:
            newimg.append(col[0])

    newimg = np.reshape(newimg, (784, 1))

    result = net.recognize(newimg)
    count = 0
    percentagesAsInts = []
    percentagesAsStrings = []
    for i in result:
        percentagesAsStrings.append('{:.0f}'.format(i[0] * 100) + "%")
        percentagesAsInts.append(int('{:.0f}'.format(i[0] * 100)))
        count += 1

    count = 0
    prediction = -1
    for digitPercentage in percentagesAsInts:
        if digitPercentage >= 50:
            prediction = count
        count += 1

    print("DEBUG: RESULT IS - " + str(prediction) + " - | ")
    if prediction == -1:
        return "Couldn't recognize :(\n" + '\n'.join(percentagesAsStrings)
    return str(prediction)


app.run()
