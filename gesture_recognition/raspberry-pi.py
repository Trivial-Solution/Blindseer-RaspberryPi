# This file is used to run the gesture recognition model on the Raspberry Pi.

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
import time
import os
import io

import base64, re
import cv2 as cv
import mediapipe as mp

from google.cloud import storage
from datetime import datetime
from picamera2 import Picamera2, Preview
from model import KeyPointClassifier


def __generate_unique_id():
    timestamp = time.time()
    dt_object = datetime.fromtimestamp(timestamp)
    unique_id = dt_object.strftime("%Y-%m-%d-%H-%M-%S")
    return unique_id


def capture_and_save_image(file_path, picam2):
    # Allow some time for the camera to adjust to light conditions
    time.sleep(3)

    # Capture the image
    picam2.capture_file("image.jpg")
    print(f"Image saved to {file_path}")


def upload_blob(bucket_name, source_file_name, destination_blob_name, gesture):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.metadata = {"gesture": gesture}
    blob.upload_from_filename(source_file_name)

    print(f"File {source_file_name} uploaded to {destination_blob_name}.")


def send_image(gesture, picam2):
    file_path = 'image.jpg'
    capture_and_save_image(file_path, picam2)

    bucket_name = 'blindseer-images'
    source_file_name = file_path  # Path to the file saved earlier
    destination_blob_name = __generate_unique_id()

    upload_blob(bucket_name, source_file_name, destination_blob_name, gesture)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.7)

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    # Initialize PiCamera2
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration())

    # Start the camera
    picam2.start()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    # Read gesture labels once before the loop starts
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    if not keypoint_classifier_labels:
        print("No labels found, exiting program.")
        return  # Exit if no labels could be read

    while True:
        image = picam2.capture_array()
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = calc_landmark_list(image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                current_gesture = keypoint_classifier_labels[hand_sign_id]

                # based on the gesture send differente metadata
                if current_gesture == "One":
                    send_image("one", picam2)
                elif current_gesture == "Two":
                    send_image("two", picam2)
                elif current_gesture == "Three":
                    send_image("three", picam2)
                elif current_gesture == "Four":
                    send_image("four", picam2)
                elif current_gesture == "Thumbs up":
                    send_image("thumbs_up", picam2)
                elif current_gesture == "Thumbs down":
                    send_image("thumbs_down", picam2)
                elif current_gesture == "Rock on":
                    send_image("rock_on", picam2)
                else:
                    break
                print(current_gesture)
                time.sleep(4)

        # Display the image in a pop-out window
        cv.imshow("Camera Feed", debug_image)

        if cv.waitKey(10) == 27:  # ESC to break
            break

    picam2.stop()
    cv.destroyAllWindows()


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'keys/skilled-mission-405818-0b879b4080fd.json'
    main()