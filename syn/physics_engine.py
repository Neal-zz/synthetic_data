import abc
import os
import time

import numpy as np
import pkg_resources
import pybullet
import trimesh
from autolab_core import Logger, RigidTransform
from pyrender import Mesh, Node, PerspectiveCamera, Scene, Viewer

class PhysicsEngine(metaclass=abc.ABCMeta):
    """Abstract Physics Engine class"""

    def __init__(self):

        # set up logger
        self._logger = Logger.get_logger(self.__class__.__name__)

    @abc.abstractmethod
    def start(self):
        pass

    @abc.abstractmethod
    def stop(self):
        pass

class PybulletPhysicsEngine(PhysicsEngine):
    """Wrapper for pybullet physics engine that is tied to a single ID"""

    def __init__(self, debug=False):
        PhysicsEngine.__init__(self)
        self._physics_client = None
        self._debug = debug

    def add(self, obj, static=False):

        if self._debug:
            self._viewer.render_lock.acquire()
            n = Node(
                mesh=Mesh.from_trimesh(obj.mesh),
                matrix=obj.pose.matrix,
                name=obj.key,
            )
            self._scene.add_node(n)
            self._viewer.render_lock.release()

    def remove(self, key):

        if self._debug:
            self._viewer.render_lock.acquire()
            if self._scene.get_nodes(name=key):
                self._scene.remove_node(
                    next(iter(self._scene.get_nodes(name=key)))
                )
            self._viewer.render_lock.release()

    def start(self):
        if self._physics_client is None:
            self._physics_client = pybullet.connect(pybullet.DIRECT)
            pybullet.setGravity(
                0, 0, -9.81, physicsClientId=self._physics_client
            )
            if self._debug:
                self._create_scene()
                self._viewer = Viewer(
                    self._scene, use_raymond_lighting=True, run_in_thread=True
                )

    def stop(self):
        if self._physics_client is not None:
            pybullet.disconnect(self._physics_client)
            self._physics_client = None
            if self._debug:
                self._scene = None
                self._viewer.close_external()
                while self._viewer.is_active:
                    pass

    def _create_scene(self):
        self._scene = Scene()
        camera = PerspectiveCamera(
            yfov=0.833, znear=0.05, zfar=30.0, aspectRatio=1.0
        )
        cn = Node()
        cn.camera = camera
        pose_m = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 2.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        pose_m[:, 1:3] *= -1.0
        cn.matrix = pose_m
        self._scene.add_node(cn)
        self._scene.main_camera_node = cn

