import face_recognition
import os
import time
import pickle
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ExifTags
from flask import Flask, request, jsonify
from flask_cors import CORS

MODEL_PATH = 'party_model.dat'
app = Flask(__name__)
CORS(app)

@app.route('/compare_by_photo', methods=['POST'])
def compare_by_photo():
    file = request.files['file']
    return jsonify(identify(extract_most_significant_face(file)))

def extract_most_significant_face(file_stream):
    pil_image = rotate_image(Image.open(file_stream))
    image = np.array(pil_image)
    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

    if not face_locations:
        print('No face')
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
    return image[top:bottom, left:right]

    
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

    faces_encodings = face_recognition.face_encodings(image)

    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)

    # predict classes and cull classifications that are not with high confidence
    return knn_clf.predict(faces_encodings)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001)