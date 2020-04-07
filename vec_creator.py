import sys
import os
import dlib
import glob
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Run a face recognizer")
parser.add_argument(
        '-sp',
        '--shape_predictor',
        type=str,
        default='models/shape_predictor_68_face_landmarks.dat',
        help='Set shape predictor path')

parser.add_argument(
        '-fr',
        '--face_recognition',
        type=str,
        default='models/dlib_face_recognition_resnet_model_v1.dat',
        help='Set face recognizer path'
    )

parser.add_argument(
        '-vp',
        '--vectors_path',
        type=str,
        default='vectors',
        help='Set vectors path.'
    )

parser.add_argument(
        '-pp',
        '--photo_path',
        type=str,
        required=True,
        help='Set face photo path'
    )

parser.add_argument(
        '-id',
        '--ID',
        type=int,
        required=True,
        help='Set ID'
    )

args = parser.parse_args()

faces_file_path = args.photo_path

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(args.shape_predictor)
facerec = dlib.face_recognition_model_v1(args.face_recognition)

img = dlib.load_rgb_image(faces_file_path)

dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))

for k, d in enumerate(dets):
    shape = sp(img, d)
    #TODO finish this shit
    print("Computing descriptor on aligned image ..")
    face_chip = dlib.get_face_chip(img, shape)        
    face_descriptor = facerec.compute_face_descriptor(face_chip)                
    vector = [x for x in face_descriptor]
    dict = {str(args.ID) : vector}
    
    json = json.dumps(dict)
    f = open(f'{args.vectors_path}/{args.ID}.json',"w")
    f.write(json)
    f.close()



