import numpy as np
import pathlib


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

P = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]]
        ).astype(np.float32)

def cam_info_from_llff(cams, downsample=1, type=1):
    """_summary_

    Args:
        cams (np.array): cams information in LLFF format
        downsample: 1.13
    """
    poses = cams[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
    # coordinate transformation OpenGL->Colmap, center poses
    H, W, focal = poses[0, :, -1]
    focal = focal / downsample
    # focal = (focal, focal)
    K = np.array([[focal, 0, W//2],
                    [0, focal, H//2],
                    [0, 0, 1]]).astype(np.float32)
    # print(K)
    poses = np.concatenate([poses[..., :1], -poses[..., 1:2], -poses[..., 2:3], poses[..., 3:4]], -1)
    # poses, _ = center_poses(poses)  # Re-center poses so that the average is near the center.
    # prepare poses
    image_poses = []
    # self.image_times = []
    for idx in range(poses.shape[0]):
        pose = poses[idx]
        # if type:
        #     pose[:3, -1] /= 1000
        c2w = np.concatenate((pose, np.array([[0, 0, 0, 1]])), axis=0) # 4x4
        c2w = P @ c2w @ P.T
        # w2c = np.linalg.inv(c2w)
        # R = w2c[:3, :3]
        # T = w2c[:3, -1]
        # R = np.transpose(R)
        image_poses.append(c2w.astype(np.float32))
        # self.image_times.append(idx / poses.shape[0])
        # print(c2w.dtype)
    return K, image_poses