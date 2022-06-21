import cv2
import time
import logging
import argparse
import numpy as np

# Logging stuff
LOG_FORMAT = "(%(levelname)s) %(asctime)s - %(message)s"
# create and configure logger
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, filemode="w")
logger = logging.getLogger()


def parse_arguments():
    """Parse input args"""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--img",
        type=str,
        default="./input.jpg",
        help="path of input image.",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=60.0,
        help="Field of view of the generated perspective image.",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=15.0,
        help="Horizontal angle in degree. Right direction is positive",
    )
    parser.add_argument(
        "--phi",
        type=float,
        default=0.0,
        help="Vertical angle in degree. Up direction is positive",
    )

    return parser.parse_args()


def xyz2lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    return out


def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / np.pi + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)

    return out


class Equirectangular:
    def __init__(self, img_name):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        [self._height, self._width, _] = self._img.shape

    def get_perspective(self, FOV, THETA, PHI, height, width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        K = np.array(
            [
                [f, 0, cx],
                [0, f, cy],
                [0, 0, 1],
            ],
            np.float32,
        )
        K_inv = np.linalg.inv(K)

        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)
        z = np.ones_like(x)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        xyz = xyz @ K_inv.T

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
        R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
        R = R2 @ R1
        xyz = xyz @ R.T
        lonlat = xyz2lonlat(xyz)
        XY = lonlat2XY(lonlat, shape=self._img.shape).astype(np.float32)
        persp = cv2.remap(
            self._img,
            XY[..., 0],
            XY[..., 1],
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_WRAP,
        )

        return persp


if __name__ == "__main__":
    start_time = time.time()

    args = parse_arguments()  # parse input args
    path_img = args.img
    fov = args.fov
    theta = args.theta
    phi = args.phi

    equi = Equirectangular(path_img)  # Load equirectangular image

    #
    # FOV unit is degree
    # theta is z-axis (horizontal) angle (right direction is positive, left direction is negative)
    # phi is y-axis (vertical) angle (up direction positive, down direction negative)
    # height and width is output image dimension
    #
    img = equi.get_perspective(
        fov, theta, phi, 720, 1080
    )  # Specify parameters(FOV, theta, phi, height, width)
    cv2.imwrite("result.jpg", img)

    elapsed_time = time.time() - start_time
    logger.info("Elapsed time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
