import numpy as np
# from calibration_utils import CalibrationUtils
from pyquaternion import Quaternion

class TransformationUtils:
    
    @staticmethod
    def transform_points(radar_points, camera_calib, radar_calib):

        xyz = radar_points[:, :3]

        ones = np.ones((xyz.shape[0], 1))
        xyz_h = np.hstack((xyz, ones))
        T_r_c = TransformationUtils.build_transformation_matrix(camera_calib, radar_calib)
        xyz_camera = (T_r_c @ xyz_h.T).T

        radar_p_c = radar_points.copy()
        radar_p_c[:, :3] = xyz_camera[:, :3]
        return radar_p_c
    
    @staticmethod
    def build_transformation_matrix(camera_calib, radar_calib):
        t_r = np.array(radar_calib["translation"]).reshape(3,1)
        R_r = Quaternion(radar_calib["rotation"]).rotation_matrix

        t_c = np.array(camera_calib["translation"]).reshape(3,1)
        R_c = Quaternion(camera_calib["rotation"]).rotation_matrix

        T_r = np.eye(4)
        T_r[:3, :3] = R_r
        T_r[:3, 3] = t_r.flatten()

        T_c = np.eye(4)
        T_c[:3, :3] = R_c
        T_c[:3, 3] = t_c.flatten()

        T_cam_radar =  np.linalg.inv(T_c) @ T_r

        return T_cam_radar


    @staticmethod
    def point_cloud_to_pixel(points_cloud, camera_calib):
        cam_intrisic = camera_calib["camera_intrinsic"]

def main():

    DATAROOT = "/home/danitur2/semida/licenta/data/nuscenes"

    nusc = NuScenes(
        version="v1.0-mini",
        dataroot=str(DATAROOT),
        verbose=False
    )

    scene = nusc.scene[0]
    sample = nusc.get("sample", scene["first_sample_token"])

    calib = CalibrationUtils(nusc)
    camera_calib = calib.get_camera_calibration(sample)
    radar_calib = calib.get_radar_calibration(sample)

