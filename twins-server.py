import face_recognition
import os
import time
import pickle
import numpy as np
from os import listdir
from os.path import isdir, join, isfile, splitext
import glob
from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ExifTags
from flask import Flask, request, jsonify
from flask_cors import CORS

MODEL_PATH = 'party_model.dat'
CLASSES_DIR = '/home/henrik/face-recognition-server/static'

app = Flask(__name__)
CORS(app)

@app.route('/compare_by_photo', methods=['POST'])
def compare_by_photo():
    file = request.files['file']
    user_id = request.form['userId']

    try:
        face_image, face_count = extract_most_significant_face(file)
    except (LookupError):
        return jsonify(status=NO_FACE)

    identified_user_id = identify(face_image)

    if (user_id != identified_user_id):
        return jsonify(status=PRANK_TRY,
            identified_user_id=identified_user_id,
            face_count=face_count)
    else:
        compare(face_image)
        return jsonify(status='OK')


def extract_most_significant_face(file_stream):
    pil_image = rotate_image(Image.open(file_stream))
    image = np.array(pil_image)
    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

    if not face_locations:
        raise LookupError('No face found')

    largest_area = 0
    selected_location = face_locations[0]
    for face_location in face_locations:
        
        top, right, bottom, left = face_location
        area = (bottom - top) * (right - left)
        if (area > largest_area):
            largest_area = area
            selected_location = face_location

    top, right, bottom, left = selected_location
    return image[top:bottom, left:right], len(face_locations)

    
def rotate_image(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif=dict(image._getexif().items())

        if exif[orientation] == 3:
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image=image.rotate(90, expand=True)

    except (AttributeError, KeyError, IndexError):
        pass
    return image


def identify(image):
    with open(MODEL_PATH, 'rb') as f:
            knn_clf = pickle.load(f)
    try:
        user_id = knn_clf.predict(face_recognition.face_encodings(image))[0]
    except:
        user_id = ''
    f.close()
    return user_id


def compare(image):
    for class_dir in listdir(CLASSES_DIR):
        model_files = glob.glob(join(train_dir, class_dir,'/model.dat'), recursive=False)
        if (model_files):
            print('OK')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001)