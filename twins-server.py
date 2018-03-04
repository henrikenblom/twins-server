import face_recognition
import os
import time
import pickle
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ExifTags
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/compare_by_photo', methods=['POST'])
def compare_by_photo():
    file = request.files['file']
    main_face_image = Image.fromarray(extract_most_significant_face(file))
    main_face_image.save('main_face.jpg', 'jpeg')
    return jsonify('ok')

def extract_most_significant_face(file_stream):
    pil_image = rotate_image(Image.open(file_stream))
    print('here')
    image = np.array(pil_image)
    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

    if not face_locations:
        print('No face')
        raise LookupError('No face found')

    if len(face_locations) == 1:
        print('Found one face')
        return image[face_locations[0]]


    
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001)