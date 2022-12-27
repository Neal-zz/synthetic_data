import os
import time

import gym
import numpy as np
import scipy.stats as sstats
import trimesh
from autolab_core import Logger, RigidTransform

from .random_variables import CameraRandomVariable
from .states import CameraState, HeapAndCameraState, HeapState, ObjectState

KEY_SEP_TOKEN = "~"

class CameraStateSpace(gym.Space):
    """State space for a camera."""

    def __init__(self, config):
        self._config = config

        # read params
        self.frame = config["name"]

        # random variable for pose of camera
        self.camera_rv = CameraRandomVariable(config)

    def sample(self, cart_pos):
        """Sample a camera state."""
        pose, intrinsics = self.camera_rv.sample(cart_pos, size=1)
        return CameraState(self.frame, pose, intrinsics)


class HeapStateSpace(gym.Space):
    """State space for object heaps."""

    def __init__(self, physics_engine, config):

        self._physics_engine = physics_engine
        self._config = config  # heap

        # set up logger
        # dataset_generation.log
        self._logger = Logger.get_logger(self.__class__.__name__)

        # read subconfigs
        obj_config = config["objects"]
        workspace_config = config["workspace"]
        if (workspace_config["width"] < 6 or workspace_config["width"] > 10 or
            workspace_config["length"] < 8):

            self._logger.warning(
                "Error room size... width: %s; length: %s".format(
                    workspace_config["width"], workspace_config["length"])
            )

        self.box_size = workspace_config["box_size"]  # 2
        self.num_objs = 7
        self.replace = config["replace"]  # 0

        # Set up object configs
        # bounds of object pose in each box
        # organized as [tx, ty, theta]
        min_obj_pose = np.r_[obj_config["planar_translation"]["min"], obj_config["planar_translation"]["min"], 0]
        max_obj_pose = np.r_[obj_config["planar_translation"]["max"], obj_config["planar_translation"]["max"], 2 * np.pi]
        self.obj_planar_pose_space = gym.spaces.Box(
            min_obj_pose, max_obj_pose, dtype=np.float32
        )
        self.trans_range = max_obj_pose[0]-min_obj_pose[0]

        self.obj_density = 4000

        # Setup target keys and directories
        target_keys = []
        target_mesh_filenames = []
        target_mesh_dir = obj_config["target_mesh_dir"]  # datasets/target
        if not os.path.isabs(target_mesh_dir):  # 更改为绝对路径
            target_mesh_dir = os.path.join(os.getcwd(), target_mesh_dir)
        for root, _, files in os.walk(target_mesh_dir):  # root：子文件夹全称，files：子文件夹内的文件列表
            dataset_name = os.path.basename(root)  # 子文件夹名称
            for f in files:
                _, ext = os.path.splitext(f)  # retunr [img, .png]
                if ext.split(".")[1] in trimesh.exchange.load.mesh_formats():  # stl is good.
                    target_keys.append(dataset_name)  # target1
                    target_mesh_filenames.append(os.path.join(root, f))

        self.all_target_keys = list(target_keys)
        self.target_ids = dict(
            [(key, i + 1) for i, key in enumerate(self.all_target_keys)]
        )
        self.target_mesh_filenames = {}
        [
            self.target_mesh_filenames.update({k: v})
            for k, v in zip(self.all_target_keys, target_mesh_filenames)
        ]

        # Setup object keys and directories
        object_keys = []
        mesh_filenames = []
        _mesh_dir = obj_config["mesh_dir"]  # datasets/objects
        if not os.path.isabs(_mesh_dir):  # 更改为绝对路径
            _mesh_dir = os.path.join(os.getcwd(), _mesh_dir)
        for root, _, files in os.walk(_mesh_dir):  # root：子文件夹全称，files：子文件夹内的文件列表
            dataset_name = os.path.basename(root)  # 子文件夹名称
            for f in files:
                _, ext = os.path.splitext(f)  # retunr [img, .png]
                if ext.split(".")[1] in trimesh.exchange.load.mesh_formats():  # stl is good.
                    object_keys.append(dataset_name)  # drip1
                    mesh_filenames.append(os.path.join(root, f))

        self.all_object_keys = list(np.array(object_keys))
        self.obj_ids = dict(
            [(key, i + 1) for i, key in enumerate(self.all_object_keys)]
        )
        self.mesh_filenames = {}
        [
            self.mesh_filenames.update({k: v})
            for k, v in zip(self.all_object_keys, mesh_filenames)
        ]

    @property
    def obj_keys(self):
        return self.all_object_keys

    def sample(self):
        """Samples a state from the space
        Returns
        -------
        :obj:`HeapState`
            state of the object pile
        """

        # Start physics engine
        self._physics_engine.start()

        """ setup workspace."""
        workspace_obj_states = []
        workspace_objs = self._config["workspace"]["objects"]  # room & ceiling
        for work_key, work_config in workspace_objs.items():

            # make paths absolute
            mesh_filename = work_config["mesh_filename"]  # datasets/room/room.stl
            pose_filename = work_config["pose_filename"]  # datasets/room/room_pose.tf
            # 更改为绝对路径
            if not os.path.isabs(mesh_filename):
                mesh_filename = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "..",
                    mesh_filename,
                )
            if not os.path.isabs(pose_filename):
                pose_filename = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "..",
                    pose_filename,
                )

            # load mesh
            mesh = trimesh.load_mesh(mesh_filename)
            mesh.visual = trimesh.visual.ColorVisuals(mesh,
                vertex_colors=np.append(np.random.random(3), np.random.random(1)/2.0))
            mesh.density = self.obj_density  # 4000
            pose = RigidTransform.load(pose_filename)
            if work_key == "ceiling":
                self.ceiling_height = 3.5 + np.random.random(1)
                pose.translation[2] = self.ceiling_height

            workspace_obj = ObjectState(
                "{}{}0".format(work_key, KEY_SEP_TOKEN), mesh, pose
            )  # room~0
            self._physics_engine.add(workspace_obj, static=True)  # 加到环境中，不做动态仿真。
            workspace_obj_states.append(workspace_obj)

        """ sample target."""
        total_num_targets = len(self.all_target_keys)
        target_id = np.random.choice(
            np.arange(total_num_targets), size=1
        )  # 0 代表不能重复取值

        # sample target's pose
        objs_in_heap = []

        # load model
        obj_key = self.all_target_keys[target_id[0]]
        obj_mesh = trimesh.load_mesh(self.target_mesh_filenames[obj_key])
        obj_mesh.visual = trimesh.visual.ColorVisuals(obj_mesh,
            vertex_colors=np.append(np.random.random(3), np.random.random(1)/2.0+0.5))
        obj_mesh.density = self.obj_density
        obj_state_key = "{}{}{}".format(
            obj_key, KEY_SEP_TOKEN, 0
        )  # target1~0
        tar = ObjectState(obj_state_key, obj_mesh)
        self._logger.info(obj_state_key)

        # sample target's planar pose
        obj_planar_pose = self.obj_planar_pose_space.sample()  # [x,y,theta]
        theta = obj_planar_pose[2]
        R_tar_world = RigidTransform.z_axis_rotation(theta)
        t_tar_world = np.array(
            [obj_planar_pose[0]/self.trans_range*self.box_size,
            obj_planar_pose[1]-self.box_size/2.0, 0.0]
        )
        tar.pose = RigidTransform(
            rotation=R_tar_world,
            translation=t_tar_world,
            from_frame="obj",
            to_frame="world",
        )

        self._physics_engine.add(tar, static=True)  # 静态
        objs_in_heap.append(tar)

        """ sample objects."""
        total_num_objs = len(self.all_object_keys)
        num_objs = self.num_objs  # = 7
        obj_inds = np.random.choice(
            np.arange(total_num_objs), size=2*num_objs, replace=self.replace
        )  # 0 代表不能重复取值

        # sample object's pose
        total_drops = 0
        while total_drops < num_objs:

            # load model
            if total_drops == 6 and t_tar_world[0] < 0:
                boxid = 8
            else:
                boxid = total_drops
            
            obj_key = self.all_object_keys[obj_inds[total_drops]]
            obj_mesh = trimesh.load_mesh(self.mesh_filenames[obj_key])
            obj_mesh.visual = trimesh.visual.ColorVisuals(obj_mesh,
                vertex_colors=np.append(np.random.random(3), np.random.random(1)/2.0+0.5))
            obj_mesh.density = self.obj_density
            obj_state_key = "{}{}{}".format(
                obj_key, KEY_SEP_TOKEN, total_drops
            )  # drip1~0
            obj = ObjectState(obj_state_key, obj_mesh)
            self._logger.info(obj_state_key)

            # sample object planar pose
            obj_planar_pose = self.obj_planar_pose_space.sample()  # [x,y,theta]
            theta = obj_planar_pose[2]
            R_obj_world = RigidTransform.z_axis_rotation(theta)
            t_obj_world = np.array(
                [obj_planar_pose[0] + (boxid%3)*self.box_size - self.box_size,
                obj_planar_pose[1] - np.floor(boxid/3)*self.box_size + 1.5*self.box_size,
                0.0]
            )
            if obj_key[:5] == "light":
                t_obj_world[2] = self.ceiling_height

            obj.pose = RigidTransform(
                rotation=R_obj_world,
                translation=t_obj_world,
                from_frame="obj",
                to_frame="world",
            )

            self._physics_engine.add(obj, static=True)  # 静态
            objs_in_heap.append(obj)
            total_drops += 1

        # Stop physics engine
        self._physics_engine.stop()

        return HeapState(workspace_obj_states, objs_in_heap, tar.pose, self.ceiling_height)


class HeapAndCameraStateSpace(gym.Space):
    """State space for environments."""

    def __init__(self, physics_engine, config):

        heap_config = config["heap"]
        cam_config = config["camera"]

        # individual state spaces
        self.heap = HeapStateSpace(physics_engine, heap_config)
        self.camera = CameraStateSpace(cam_config)

    @property
    def obj_keys(self):
        return self.heap.obj_keys

    def sample(self):
        """Sample a state."""
        # sample individual states
        heap_state = self.heap.sample()
        cam_state = self.camera.sample(heap_state.cart_pose.translation)

        return HeapAndCameraState(heap_state, cam_state)
