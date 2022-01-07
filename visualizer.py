import cv2
import argparse
from webcam import Webcam
from threading import Thread
import time
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression

class CamVisualizer():

    def __init__(self, ip_address, user, password):
        self.last_command = 'stop'
        self.people_detection = False
        self.detection_method = 'Hog'
        self.webcam = Webcam(ip_address, user, password)
        self.webcam.start()
        cv2.startWindowThread()
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.pedestrian_cascade = cv2.CascadeClassifier(
            'Classifiers/haarcascade_fullbody.xml')

    def _key_press_event(self, key_pressed):
        if key_pressed == ord('w'):
            self.webcam.move_direction('up')
            self.last_command = 'rotate'
        elif key_pressed == ord('s'):
            self.webcam.move_direction('down')
            self.last_command = 'rotate'
        elif key_pressed == ord('a'):
            self.webcam.move_direction('left')
            self.last_command = 'rotate'
        elif key_pressed == ord('d'):
            self.webcam.move_direction('right')
            self.last_command = 'rotate'
        elif key_pressed == ord('p'):
            self.people_detection = not self.people_detection
        elif key_pressed == ord('m'):
            if self.detection_method == 'Hog':
                self.detection_method = 'Cascade'
            else:
                self.detection_method = 'Hog'
        elif key_pressed in range(49, 58):
            self.webcam.call_preset(key_pressed)
            self.last_command = 'stop'          
        else:
            if self.last_command != 'stop':
                time.sleep(0.05)
                self.webcam.move_direction('stop')
                self.last_command = 'stop'   

    def _people_detector(self, frame):
        max_width = 600
        width = frame.shape[1]
        if width > max_width:
            frame = imutils.resize(frame, width=max_width)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        if self.detection_method == 'Hog':
            boxes, weights = self.hog.detectMultiScale(gray, winStride=(16,16), scale=1.05)
            boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
            pedestrians = non_max_suppression(boxes, probs=None, overlapThresh=0.65)
        else:
            pedestrians = self.pedestrian_cascade.detectMultiScale(gray, 1.1, 1)
            weights = [0 for a in pedestrians]
        
        for a, weight in zip(pedestrians, weights):
            xA, yA, xB, yB = a
            # display the detected boxes in the colour picture
            cv2.putText(frame, f'P{weight}', (xA, yA), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 2)
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
            
        return frame

    def start_visualization(self):
        while True:
            frame = cv2.resize(self.webcam.get_current_frame(),
                        (640, 480))
            frame = self.webcam.get_current_frame()
            
            if self.people_detection:
                frame = self._people_detector(frame)
            
            cv2.imshow('CameraIP', frame)
            
            key_pressed = cv2.waitKey(50) & 0xFF
            
            #if user presses q quit program
            if key_pressed  == ord('q'):
                break  
            else:
                self._key_press_event(key_pressed)
        
        self.webcam.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--ip_address', required=True,
	help='Ip Adress To Camera')
    ap.add_argument('-u', '--user', required=True,
	help='Cam User')
    ap.add_argument('-p', '--password', required=True,
	help='Camera Password')
    args = vars(ap.parse_args())
    
    vis = CamVisualizer(args['ip_address'], args['user'], args['password'])
    
    try:
        vis.start_visualization()
    except KeyboardInterrupt:
        vis.webcam.stop()
        cv2.destroyAllWindows()
    except Exception as e:
        vis.webcam.stop()
        cv2.destroyAllWindows()
        raise e
