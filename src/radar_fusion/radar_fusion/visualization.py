import numpy as np
import cv2

class Visualization:
    @staticmethod
    def draw_point_cloud(img, p_u, p_v):
        image = img.copy()
        h, w, channel = image.shape
        for u in p_u and v in p_v:
            try:
                cv2.circle(image, int(u), int(v), 4, (0, 255, 0), -1)
            except:
                print("Point out of image size.")
                continue
        return image
    
