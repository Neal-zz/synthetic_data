import numpy as np


class State(object):
    """Abstract class for states"""

    pass


class ObjectState(State):
    """The state of an object
    Attributes
    ----------
    key : str
        string identifying the object
    mesh : Trimesh
        stores geometry of the object
    pose : RigidTransform
        describes the pose of the object in the world
    sim_id : int
        id for the object in sim
    """

    def __init__(self, key, mesh, pose=None, sim_id=-1):
        self.key = key
        self.mesh = mesh
        self.pose = pose
        self.sim_id = sim_id

    @property
    def center_of_mass(self):
        return self.mesh.center_mass

    @property
    def density(self):
        return self.mesh.density


class HeapState(State):
    """State of a set of objects in a heap.

    Attributes
    ----------
    obj_states : list of ObjectState
        state of all objects in a heap
    """

    def __init__(self, workspace_states, obj_states, metadata={}):
        self.workspace_states = workspace_states
        self.obj_states = obj_states
        self.metadata = metadata

    @property
    def workspace_keys(self):
        return [s.key for s in self.workspace_states]

    @property
    def workspace_meshes(self):
        return [s.mesh for s in self.workspace_states]

    @property
    def workspace_sim_ids(self):
        return [s.sim_id for s in self.workspace_states]

    @property
    def obj_keys(self):
        return [s.key for s in self.obj_states]

    @property
    def obj_meshes(self):
        return [s.mesh for s in self.obj_states]

    @property
    def obj_sim_ids(self):
        return [s.sim_id for s in self.obj_states]

    @property
    def num_objs(self):
        return len(self.obj_keys)

    def __getitem__(self, key):
        return self.state(key)

    def state(self, key):
        try:
            return self.obj_states[self.obj_keys.index(key)]
        except:
            try:
                return self.workspace_states[self.workspace_keys.index(key)]
            except:
                logging.warning("Object %s not in pile!")
        return None


class CameraState(State):
    """State of a camera.
    Attributes
    ----------
    mesh : Trimesh
        triangular mesh representation of object geometry
    pose : RigidTransform
        pose of camera with respect to the world
    intrinsics : CameraIntrinsics
        intrinsics of the camera in the perspective projection model.
    """

    def __init__(self, frame, pose, intrinsics):
        self.frame = frame
        self.pose = pose
        self.intrinsics = intrinsics

    @property
    def height(self):
        return self.intrinsics.height

    @property
    def width(self):
        return self.intrinsics.width

    @property
    def aspect_ratio(self):
        return self.width / float(self.height)

    @property
    def yfov(self):
        return 2.0 * np.arctan(self.height / (2.0 * self.intrinsics.fy))


class HeapAndCameraState(object):
    """State of a heap and camera."""

    def __init__(self, heap_state, cam_state):
        self.heap = heap_state
        self.camera = cam_state

    @property
    def obj_keys(self):
        return self.heap.obj_keys

    @property
    def num_objs(self):
        return self.heap.num_objs
