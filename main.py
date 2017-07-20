import cv2
import numpy as np
from os import path
import pywt
from saliency import Saliency
from tracking import MultipleObjectsTracker


def main(video_file= 'jet.mp4', region=((0, 0), (350, 450))):
    # open video file
    
    if path.isfile(video_file):
        video = cv2.VideoCapture(video_file)
    else:
        print 'File "' + video_file + '" does not exist.'
        raise SystemExit

    # initialize tracker
    mot = MultipleObjectsTracker()

    while True:
        # grab next frame
        success, img = video.read()
        if success:
            if region:
                #original video is too big: grab some meaningful region of interest
                img = img[region[0][0]:region[1][0], region[0][1]:region[1][1]]

            # generate saliency map
            sal = Saliency(img)

            cv2.imshow('original', img)
            cv2.imshow('saliency', sal.get_saliency_map())
            cv2.imshow('objects', sal.get_proto_objects_map())
            cv2.imshow('tracker', mot.advance_frame(img,
                       sal.get_proto_objects_map()))

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        else:
            break


if __name__ == '__main__':
    main()
