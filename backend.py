import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from keras.models import load_model
from shutil import copy
import tensorflow as tf


UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG', 'gif', 'GIF'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

global model
model = load_model("underage_detector_02.h5")
global graph
graph = tf.get_default_graph()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            file.save(file_path)

            copy(file_path, 'static/'+filename)

            file_path = 'static/'+filename
            return redirect(url_for('detection', path=filename))

    return render_template("index.html")


@app.route('/<path>')
def detection(path):

    file_path = 'static/'+path
    # file_path = 'Dataset/6-20/' + path

    # print(file_path)

    img = cv2.imread(file_path)
    # print(type(img))

    if img.shape[0] == 200 and img.shape[1] == 200:
        X = []
        X.append(img)
        X = np.array(X)
        X = X.astype('float32')
        X /= 255

        # X = convert_to_tensor(X)

        with graph.as_default():
            pred_y = model.predict(X)

        # 0 is underage
        if pred_y[0][0] > pred_y[0][1]:
           label = 'underage'
        else:
            label = 'adult'

        return render_template("result.html", label=label, file_path=file_path)
    else:
        return "The uploaded image's height and width must be 200 and 200."


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=False)
    # app.run(port=80, debug=True)

    # e = 0
    # img_count = 0
    # for img in os.listdir('Dataset/6-20/'):
    #     img_count += 1
    #     label = detection(img)
    #     if label == 'Adult':
    #         e += 1
    #     if img_count >= 100:
    #         break
    #
    # print(e)
