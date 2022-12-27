import numpy as np
import scipy.stats as sstats
from autolab_core import CameraIntrinsics, RigidTransform, transformations
from autolab_core.utils import sph2cart


class CameraRandomVariable(object):
    """Uniform distribution over camera poses and intrinsics about a viewsphere over a planar worksurface.
    The camera is positioned pointing towards (0,0,0).
    """

    def __init__(self, config):

        # read params
        self.config = config
        self._parse_config(config)

        # viewsphere
        self.rad_rv = sstats.uniform(
            loc=self.min_radius, scale=self.max_radius - self.min_radius
        )
        self.hei_rv = sstats.uniform(
            loc=self.min_height, scale=self.max_height - self.min_height
        )
        self.the_rv = sstats.uniform(
            loc=self.min_theta, scale=self.max_theta - self.min_theta
        )

        # table translation
        self.tx_rv = sstats.uniform(
            loc=self.min_x, scale=self.max_x - self.min_x
        )

    def _parse_config(self, config):
        """Reads parameters from the config into class members."""
        # camera params
        self.frame = config["name"]                 # camera
        self.focal_length = config["focal_length"]  # 365
        self.im_height = config["im_height"]        # 512
        self.im_width = config["im_width"]          # 512
        self.mean_cx = float(self.im_width - 1) / 2.0
        self.mean_cy = float(self.im_height - 1) / 2.0
        self.threshold = config["threshold"]        # 4

        # viewsphere params
        self.min_radius = config["radius"]["min"]  # 距离
        self.max_radius = config["radius"]["max"]
        self.min_height = config["height"]["min"]
        self.max_height = config["height"]["max"]
        self.min_theta = np.deg2rad(config["theta"]["min"])
        self.max_theta = np.deg2rad(config["theta"]["max"])

        # params of translation in plane
        self.min_x = config["x"]["min"]
        self.max_x = config["x"]["max"]

    def camera_to_world_pose(self, cart_pos, radius, height, theta, x):
        """Convert spherical coords to a camera pose in the world."""
        # generate camera center from spherical coords
        delta_t = np.array([x*np.cos(theta) + cart_pos[0],
            x*np.sin(theta) + cart_pos[1], height])
        camera_z = np.array([radius*np.sin(theta),
            -radius*np.cos(theta), 0])
        camera_center = camera_z + delta_t
        camera_center[0] = np.min((np.max((camera_center[0], -self.threshold)),
            self.threshold))
        camera_z = -camera_z / np.linalg.norm(camera_z)

        # find the canonical camera x and y axes
        camera_x = np.array([camera_z[1], -camera_z[0], 0])
        camera_x = camera_x / np.linalg.norm(camera_x)
        camera_y = np.cross(camera_z, camera_x)
        camera_y = camera_y / np.linalg.norm(camera_y)

        # get w^T_cam
        R = np.vstack((camera_x, camera_y, camera_z)).T
        T_camera_world = RigidTransform(
            R, camera_center, from_frame=self.frame, to_frame="world"
        )

        return T_camera_world

    def sample(self, cart_pos, size=1):
        """Sample random variables from the model.
        Parameters
        ----------
        cart_pos: [x, y, 0]
            position of the cart.
        size : int
            number of sample to take
        Returns
        -------
        :obj:`list` of :obj:`CameraSample`
            sampled camera intrinsics and poses
        """
        samples = []
        for i in range(size):
            # sample camera params
            focal = self.focal_length  # 365
            cx = self.mean_cx          # 255.5?
            cy = self.mean_cy          # 255.5?

            # sample viewsphere params
            radius = self.rad_rv.rvs(size=1)[0]
            height = self.hei_rv.rvs(size=1)[0]
            theta = self.the_rv.rvs(size=1)[0]

            # sample plane translation
            tx = self.tx_rv.rvs(size=1)[0]

            # convert to pose and intrinsics
            pose = self.camera_to_world_pose(cart_pos, radius, height, theta, tx)
            intrinsics = CameraIntrinsics(
                self.frame,
                fx=focal,
                fy=focal,
                cx=cx,
                cy=cy,
                skew=0.0,
                height=self.im_height,
                width=self.im_width,
            )

            # convert to camera pose
            samples.append((pose, intrinsics))

        # not a list if only 1 sample
        if size == 1:
            return samples[0]
        return samples
