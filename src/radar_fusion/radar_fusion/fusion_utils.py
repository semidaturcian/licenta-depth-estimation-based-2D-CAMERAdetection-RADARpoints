import numpy as np

def depth_estimation(bbox, p_u, p_v, cam_points):
        distances = []
        for bb_index, bb in enumerate(bbox):
            x_min = bb.center_x - bb.width/2
            x_max = bb.center_x + bb.width/2
            y_min = bb.center_y - bb.height/2
            y_max = bb.center_y + bb.height/2
            dist_points = []
            for index, (u, v) in enumerate(zip(p_u, p_v)):
                if (x_min <= u <= x_max) and (y_min <= v <= y_max):
                    point_camera = cam_points[index]
                    # X = point_camera[0]
                    # Y = point_camera[1]
                    Z = point_camera[2]
                    dist = np.mean(Z)
                    dist_points.append(dist)
                    print(
                    f"BB {bb_index}: "
                    f"Radar point {index} -> "
                    f"pixel=({u:.1f}, {v:.1f}), "
                    f"Z={Z:.2f} m"
                        )
            distance = min(dist_points) if dist_points else np.inf
            print(
                f"BB {bb_index}: "
                f"{len(dist_points)} radar points -> "
                f"distance={distance:.2f} m"
                 )
        distances.append(distance)
        return distances

          
            