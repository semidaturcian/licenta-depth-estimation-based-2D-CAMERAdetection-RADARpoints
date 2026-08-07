import numpy as np
import cv2

class Visualization:
    @staticmethod
    def draw_detection(img, bbox):
        image = img.copy()
        for bb in bbox:
            x1 = int(bb.center_x - bb.width/2)
            y1 = int(bb.center_y - bb.height/2)

            x2 = int(bb.center_x + bb.width/2)
            y2 = int(bb.center_y + bb.height/2)
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return image
    
