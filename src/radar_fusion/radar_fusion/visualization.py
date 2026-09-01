import numpy as np
import cv2

class Visualization:
    @staticmethod
    def draw_point_cloud(img, p_u, p_v):
        image = img.copy()
        h, w, _ = image.shape
        for u, v in zip(p_u, p_v):
            if not (0 <= u < w and 0 <= v < h):
                continue
            cv2.circle(image, (int(u), int(v)), 4, (255, 0, 0), -1)          
        return image

    @staticmethod
    def draw_detection(img, distances_based_median, bbox, p_u, p_v):
        image = img.copy()
        for indx, bb in enumerate(bbox):
            x1 = int(bb.center_x - bb.width/2)
            y1 = int(bb.center_y - bb.height/2)

            x2 = int(bb.center_x + bb.width/2)
            y2 = int(bb.center_y + bb.height/2)
            dist = distances_based_median[indx]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"dist: {dist:.2f} m"
            cv2.putText(image,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
            for u, v in zip(p_u, p_v):
                        if  (x1 <= u <= x2 and y1 <= v <= y2):
                            cv2.circle(image, (int(u), int(v)), 4, (0, 0, 255), -1)
            
        return image
    
