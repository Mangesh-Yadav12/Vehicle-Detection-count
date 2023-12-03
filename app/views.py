# importing library
import cv2
from app import app
from flask import render_template, request
import numpy as np
import random
import string
import os
from PIL import Image

# Adding path to config
app.config["INITIAL_FILE_UPLOADS"] = 'C:/Users/91952/PycharmProjects/Vehicles detection/app/static/uploads'

car_cascade_src = "C:/Users/91952/PycharmProjects/Vehicles detection/app/static/cascade/cars.xml"
bus_cascade_src = "C:/Users/91952/PycharmProjects/Vehicles detection/app/static/cascade/Bus_front.xml"

# Route to home page
@app.route("/", methods=['GET', 'POST'])
def index():
    # Execute if request is get
    if request.method == 'GET':
        full_filename = "C:/Users/91952/PycharmProjects/Vehicles detection/app/static/images/white_bg.jpg"
        return render_template('index.html', full_filename=full_filename)

    # Execute if request is post
    if request.method == 'POST':
        image_upload = request.files['image_upload']
        imagename = image_upload.filename

        # Generating unique name to save image
        letters = string.ascii_lowercase
        name = ''.join(random.choice(letters) for i in range(10)) + '.png'
        full_filename = 'uploads/' + name

        image = Image.open(image_upload)
        image = image.resize((450, 250))
        image_arr = np.array(image)
        grey = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)

        # Cascade
        car_cascade = cv2.CascadeClassifier(car_cascade_src)
        cars = car_cascade.detectMultiScale(grey, 1.1, 1)

        bcnt = 0
        bus_cascade = cv2.CascadeClassifier(bus_cascade_src)
        bus = bus_cascade.detectMultiScale(grey, 1.1, 1)
        for (x, y, w, h) in bus:
            cv2.rectangle(image_arr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            bcnt += 1

        ccnt = 0
        for (x, y, w, h) in cars:
            cv2.rectangle(image_arr, (x, y), (x + w, y + h), (255, 0, 0), 2)
            ccnt += 1

        img = Image.fromarray(image_arr, 'RGB')
        img.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], name))

        # Returning template file extracted text
        result = str(ccnt) + ' cars and ' + str(bcnt) + ' buses found'
        return render_template('index.html', full_filename=full_filename, pred=result)

# Run the application
app.run(debug=True)
