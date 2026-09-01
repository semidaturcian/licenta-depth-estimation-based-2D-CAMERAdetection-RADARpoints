import numpy as np

def depth_estimation(bbox, p_u, p_v, cam_points):

    distances_based_mean = []
    distances_based_median = []
    distances_based_min = []

    for bb_index, bb in enumerate(bbox):
        x_min = bb.center_x - bb.width / 2
        x_max = bb.center_x + bb.width / 2

        y_min = bb.center_y - bb.height / 2
        y_max = bb.center_y + bb.height / 2
        associated_points = []
        for index, (u, v) in enumerate(zip(p_u, p_v)):
            if (x_min <= u <= x_max) and (y_min <= v <= y_max):
                point_camera = cam_points[index]
                X = point_camera[0]
                Y = point_camera[1]
                Z = point_camera[2]
                associated_points.append([
                    X,
                    Y,
                    Z
                ])
                print(
                    f"BB {bb_index}: "
                    f"Radar point {index} -> "
                    f"pixel=({u:.1f}, {v:.1f}), "
                    f"X={X:.2f}, "
                    f"Y={Y:.2f}, "
                    f"Z={Z:.2f} m"
                )
        if not associated_points:
            distances_based_mean.append(np.inf)
            distances_based_median.append(np.inf)
            continue
        associated_points = np.asarray(
            associated_points,
            dtype=np.float64
        )
        # -------------------------
        # Filter outliers using Z
        # -------------------------
        filtered_points = filter_points(
            associated_points
        )
        if len(filtered_points) == 0:
            distances_based_mean.append(np.inf)
            distances_based_median.append(np.inf)
            distances_based_min.append(np.inf)
            continue
        # -------------------------
        # Min
        # -------------------------
        X_min = np.mean(filtered_points[:, 0])
        Y_min = np.mean(filtered_points[:, 1])
        Z_min = np.mean(filtered_points[:, 2])

        distance_min = distance_estimation(
            X_min,
            Y_min,
            Z_min,
            type="min"
        )
        # -------------------------
        # Mean
        # -------------------------
        X_mean = np.mean(filtered_points[:, 0])
        Y_mean = np.mean(filtered_points[:, 1])
        Z_mean = np.mean(filtered_points[:, 2])

        distance_mean = distance_estimation(
            X_mean,
            Y_mean,
            Z_mean,
            type="mean"
        )
        # -------------------------
        # Median
        # -------------------------
        X_median = np.median(filtered_points[:, 0])
        Y_median = np.median(filtered_points[:, 1])
        Z_median = np.median(filtered_points[:, 2])
        distance_median = distance_estimation(
            X_median,
            Y_median,
            Z_median,
            type="median"
        )
        print(
            f"BB {bb_index}: "
            f"{len(associated_points)} associated points -> "
            f"{len(filtered_points)} inliers"
        )
        print(
            f"Mean position: "
            f"X={X_mean:.2f}, "
            f"Y={Y_mean:.2f}, "
            f"Z={Z_mean:.2f}"
        )        
        print(
            f"Min position: "
            f"X={X_min:.2f}, "
            f"Y={Y_min:.2f}, "
            f"Z={Z_min:.2f}"
        )
        print(
            f"Median position: "
            f"X={X_median:.2f}, "
            f"Y={Y_median:.2f}, "
            f"Z={Z_median:.2f}"
        )
        print(
            f"Distance mean = {distance_mean:.2f} m"
        )

        print(
            f"Distance median = {distance_median:.2f} m"
        )
        print(
            f"Distance min = {distance_min:.2f} m"
        )
        distances_based_mean.append(distance_mean)
        distances_based_median.append(distance_median)
        distances_based_min.append(distance_min)

    return distances_based_mean, distances_based_median, distances_based_min

def filter_points(points):
    points = np.asarray(
        points,
        dtype=np.float64
    )

    if len(points) == 0:
        return np.empty((0, 3))
    z_values = points[:, 2]
    z_median = np.median(z_values)
    threshold = 3.0
    mask = np.abs(z_values - z_median) < threshold
    return points[mask]

def distance_estimation(x, y, z,  type = "mean"):
     distance = np.sqrt(x**2 + y**2 + z**2)
     print(f"Distanta {type} este {distance}")
     return distance

