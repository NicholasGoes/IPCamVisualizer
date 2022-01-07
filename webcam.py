import cv2
from threading import Thread
import requests
from copy import deepcopy
  
class Webcam:
  
    def __init__(self, ip_address, user, password):
        # Using video stream from IP Camera
        self.ip_address = ip_address
        self.user = user
        self.password = password
        self.base_url = f"http://{self.user}:{self.password}@{self.ip_address}"
        
        # This address could be different depending on camera model
        self.video_capture = cv2.VideoCapture(f"{self.base_url}/video.cgi")
        self.current_frame = self.video_capture.read()[1]
        self.last_frame = deepcopy(self.current_frame)
        self.stopThread = False

    # Creating thread for capturing images
    def start(self):
        Thread(target=self._update_frame, args=()).start()
    
    def stop(self):
        self.stopThread = True
        self.video_capture.release()
        cv2.destroyAllWindows()
        
    def _update_frame(self):
        while(not self.stopThread):
            try:
                self.last_frame = deepcopy(self.current_frame)
                self.current_frame = self.video_capture.read()[1]
            except:
                pass
        
    def call_preset(self, preset):
        preset_dict = {'49': '31',
                       '50': '33',
                       '51': '35',
                       '52': '37',
                       '53': '39'}
        preset_code = preset_dict.get(str(preset), '31')
        print(f'Calling preset {preset_code} corresponding to {preset}')
        
        requests.get(f'{self.base_url}/decoder_control.cgi?command={preset_code}', 
                     auth=(self.user, self.password))
            
    def move_direction(self, direction):
        dir_dict = {'down': '0',
                    'stop': '1',
                    'up': '2',
                    'left': '4',
                    'right': '6'}
        dir_code = dir_dict.get(direction, '1')
        print(f'Moving to direction: {direction}. Corresponding to code {dir_code}')
        
        Thread(target=self._move, args=dir_code).start()
    
    def _move(self, dir_code):
        requests.get(f'{self.base_url}/decoder_control.cgi?command={dir_code}',
                     auth=(self.user, self.password))
                
    # get the current frame
    def get_current_frame(self):
        return self.current_frame
    
    # get the last frame
    def get_last_frame(self):
        return self.last_frame


