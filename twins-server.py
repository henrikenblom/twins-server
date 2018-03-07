import face_recognition
import os
import time
import math
import pickle
import numpy as np
from os import listdir
from os.path import isdir, join, isfile, splitext
import glob
from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ExifTags
from flask import Flask, request, jsonify
from flask_cors import CORS

MODEL_PATH = 'party_model.dat'
CLASSES_DIR = 'static'
ORIGINAL_CONSTRAINTS = (1920, 1920)
DIST_THRESHOLD = 0.5
SAVE_THRESHOLD = 0.3
MARGIN = 100


app = Flask(__name__)
CORS(app)

@app.route('/compare_by_photo', methods=['POST'])
def compare_by_photo():
    file = request.files['file']
    user_id = request.form['userId']

    try:
        face_image, face_count = extract_most_significant_face(file)
    except (LookupError):
        return jsonify(status='NO_FACE')

    identified_user_id = identify(face_image)

    if (identified_user_id == ''):
        return jsonify(status='NO_FULL_FACE')
    elif (user_id != identified_user_id):
        return jsonify(status='PRANK_TRY',
            identified_user_id=identified_user_id,
            face_count=face_count)
    else:
        twin_id, closest_distance = compare(face_image, user_id)
        likeness = int(math.log1p(1 - closest_distance) * 154)
        return jsonify(status='OK',
            face_count=face_count,
            twin_id=twin_id,
            likeness=likeness)


def extract_most_significant_face(file_stream):
    pil_image = rotate_image(Image.open(file_stream))
    pil_image.thumbnail(ORIGINAL_CONSTRAINTS)
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
    bottom += (MARGIN * 2)
    right += MARGIN
    left -= MARGIN
    top -= MARGIN

    if bottom > pil_image.height:
        bottom = pil_image.height
    if right > pil_image.width:
        right = pil_image.width
    if left < 0:
        left = 0
    if top < 0:
        top = 0

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
    with open(join(CLASSES_DIR, MODEL_PATH), 'rb') as f:
            knn_clf = pickle.load(f)
    try:
        faces_encodings = face_recognition.face_encodings(image)
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        if (closest_distances[0][0][0] <= DIST_THRESHOLD):
            user_id = knn_clf.predict(faces_encodings)[0]
            if (closest_distances[0][0][0] <= SAVE_THRESHOLD):
                save_for_training(image, user_id)
        else:
            user_id = ''
    except:
        user_id = ''
    f.close()
    return user_id


def compare(image, user_id):
    classes = []
    models = []

    for class_dir in listdir(CLASSES_DIR):
        model_files = glob.glob(join(CLASSES_DIR, class_dir, 'model.dat'), recursive=False)
        if (model_files):
            with open(model_files[0], 'rb') as f:
                model = pickle.load(f)
                models.append(model)
                classes.append(class_dir)
                f.close()

    current_face_model = face_recognition.face_encodings(image)[0]
    face_distances = face_recognition.face_distance(models, current_face_model)

    closest_distance = 10
    twin_id = ''
    for i, face_distance in enumerate(face_distances):
        if (classes[i] != user_id and face_distance < closest_distance):
            twin_id = classes[i]
            closest_distance = face_distance

    return twin_id, closest_distance


def save_for_training(image, user_id):
    file_name = "{}/{}/{}.jpg".format(CLASSES_DIR, user_id, int(time.time() * 1000))
    pil_image = Image.fromarray(image)
    pil_image.save(file_name, 'jpeg')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001)