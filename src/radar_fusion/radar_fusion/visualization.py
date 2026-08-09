import numpy as np
import cv2

class Visualization:
    @staticmethod
    def draw_point_cloud(img, p_u, p_v):
        image = img.copy()
        h, w, _ = image.shape

        for u, v in zip(p_u, p_v):
            # print(f" u = {u} \n v = {v}")
            if not (0 <= u < w and 0 <= v < h):
                continue
            cv2.circle(image, (int(u), int(v)), 4, (255, 0, 0), -1)
            
        return image
    
