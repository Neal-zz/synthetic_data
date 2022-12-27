import gym
import numpy as np
from pyrender import (
    DirectionalLight,
    IntrinsicsCamera,
    Mesh,
    MetallicRoughnessMaterial,
    Node,
    OffscreenRenderer,
    RenderFlags,
    Scene,
    Viewer,
)

from .physics_engine import PybulletPhysicsEngine
from .state_spaces import HeapAndCameraStateSpace


class BinHeapEnv(gym.Env):
    """OpenAI Gym-style environment for creating object heaps in a bin."""

    def __init__(self, config):

        # read configs
        self._config = config  # (generate_mask_dataset.yaml)
        self._state_space_config = self._config["state_space"]

        # initialize variables
        self._state = None
        self._scene = None
        self._physics_engine = PybulletPhysicsEngine(
            debug=config["debug"]
        )  # datasets/objects/urdf/cache/
        self._state_space = HeapAndCameraStateSpace(
            self._physics_engine, self._state_space_config
        )

    # @property
    # def config(self):
    #     return self._config

    @property
    def state(self):
        return self._state

    @property
    def camera(self):
        return self._camera

    @property
    def obj_keys(self):
        return self.state.obj_keys

    def _reset_state_space(self):
        """Sample a new static and dynamic state."""
        state = self._state_space.sample()
        self._state = state.heap
        self._camera = state.camera

    def _update_scene(self):
        # update camera
        camera = IntrinsicsCamera(
            self.camera.intrinsics.fx,
            self.camera.intrinsics.fy,
            self.camera.intrinsics.cx,
            self.camera.intrinsics.cy,
        )
        cn = next(iter(self._scene.get_nodes(name=self.camera.frame)))
        cn.camera = camera
        pose_m = self.camera.pose.matrix.copy()
        pose_m[:, 1:3] *= -1.0
        cn.matrix = pose_m
        self._scene.main_camera_node = cn

        # update workspace
        for obj_key in self.state.workspace_keys:
            next(
                iter(self._scene.get_nodes(name=obj_key))
            ).matrix = self.state[obj_key].pose.matrix

        # update object
        for obj_key in self.state.obj_keys:
            next(
                iter(self._scene.get_nodes(name=obj_key))
            ).matrix = self.state[obj_key].pose.matrix

    def _reset_scene(self, scale_factor=1.0):
        """Resets the scene.

        Parameters
        ----------
        scale_factor : float
            optional scale factor to apply to the image dimensions
        """
        # delete scene
        if self._scene is not None:
            self._scene.clear()
            del self._scene

        # create scene
        scene = Scene()

        # setup camera
        camera = IntrinsicsCamera(
            self.camera.intrinsics.fx,
            self.camera.intrinsics.fy,
            self.camera.intrinsics.cx,
            self.camera.intrinsics.cy,
        )
        pose_m = self.camera.pose.matrix.copy()
        pose_m[:, 1:3] *= -1.0
        scene.add(camera, pose=pose_m, name=self.camera.frame)
        scene.main_camera_node = next(
            iter(scene.get_nodes(name=self.camera.frame))
        )

        # add workspace objects
        # ['room~0', 'ceiling~0']
        for obj_key in self.state.workspace_keys:

            material = MetallicRoughnessMaterial(
                baseColorFactor=np.append(np.random.random(3), 1.0),
                metallicFactor=0.2,
                roughnessFactor=0.8,
            )

            obj_state = self.state[obj_key]
            obj_mesh = Mesh.from_trimesh(obj_state.mesh, material=material)
            T_obj_world = obj_state.pose.matrix
            scene.add(obj_mesh, pose=T_obj_world, name=obj_key)

        # add scene objects
        # ['drip1~1', ...]
        for obj_key in self.state.obj_keys:
            
            material = MetallicRoughnessMaterial(
                baseColorFactor=np.append(np.random.random(3), 1.0),
                metallicFactor=0.2,
                roughnessFactor=0.8,
            )

            obj_state = self.state[obj_key]
            # smooth 设为 False，渲染图片就不再面片化。
            obj_mesh = Mesh.from_trimesh(obj_state.mesh, material=material, smooth=False)
            T_obj_world = obj_state.pose.matrix
            scene.add(obj_mesh, pose=T_obj_world, name=obj_key)

        # add light (for color rendering)
        light = DirectionalLight(color=np.ones(3), intensity=1)
        light_pose = np.eye(4)
        light_pose[1,3] = 3.0
        light_pose[2,3] = 3.0
        scene.add(light, pose=light_pose)
        ray_light_nodes = self._create_raymond_lights()
        [scene.add_node(rln) for rln in ray_light_nodes]

        self._scene = scene

    def reset_camera(self):
        """Resets only the camera.
        Useful for generating image data for multiple camera views
        """
        self._camera = self._state_space.camera.sample(self._state.cart_pose.translation)
        self._update_scene()

        return self._camera.pose

    def reset(self):
        """Reset the environment."""

        # reset state space
        self._reset_state_space()

        # reset scene
        self._reset_scene()

    def render_camera_image(self, color=True):
        """Render the camera image for the current scene."""
        renderer = OffscreenRenderer(self.camera.width, self.camera.height)
        flags = RenderFlags.NONE if color else RenderFlags.DEPTH_ONLY
        image = renderer.render(self._scene, flags=flags)
        renderer.delete()

        return image

    def _create_raymond_lights(self):
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            nodes.append(
                Node(
                    light=DirectionalLight(color=np.ones(3), intensity=1.0),
                    matrix=matrix,
                )
            )

        return nodes
