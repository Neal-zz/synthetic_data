import argparse
import gc  # 清除内存，尽量避免使用
import os
import traceback
import time
import sys
import autolab_core.utils as utils
import matplotlib.pyplot as plt
import numpy as np
from autolab_core import (
    BinaryImage,
    ColorImage,
    DepthImage,
    GrayscaleImage,
    Logger,
    TensorDataset,
    YamlConfig,
)
import open3d as o3d

#os.environ["PYOPENGL_PLATFORM"] = "osmesa"
#os.environ['PYOPENGL_PLATFORM'] = 'egl'  # pyrender backend.

sys.path.append(os.getcwd())  # /home/neal/projects/synthetic_data, used for input syn module.
from syn.bin_heap_env import BinHeapEnv

SEED = 744

# set up logger
logger = Logger.get_logger("operating_room.py")

def depth2xyz(depth_map, main_cam_pose, cur_cam_pose,
    fx, fy, cx, cy, down_sample, rm_x, rm_y, rm_z):
    
    h, w = np.mgrid[0:depth_map.shape[0],0:depth_map.shape[1]]
    # add some noise.
    z = (depth_map - np.ones([depth_map.shape[0], depth_map.shape[1]])*0.02 +
        np.random.random([depth_map.shape[0], depth_map.shape[1]])*0.04)
    x = (w-cx)*z/fx
    y = (h-cy)*z/fy
    xyz = np.dstack((x,y,z)).reshape(-1,3)

    # remove -depth point (empty points)
    thresh = 0.04
    xyz = xyz[~( xyz[:,2]<thresh )]

    # down sample. 516x516 = 266K
    pcd_num = int(depth_map.shape[0] * depth_map.shape[1] / down_sample)
    xyz = xyz[np.random.choice(xyz.shape[0], pcd_num, replace=False), :]

    # if it is not the first camera view
    if np.linalg.norm(cur_cam_pose.translation-main_cam_pose.translation) > 0.001:
        
        # translate point clouds to the world view
        xyz = (np.dot(cur_cam_pose.rotation, xyz.T).T + 
            np.array([cur_cam_pose.translation]))
        
        # remove the floor
        xyz = xyz[~( xyz[:,2]<(thresh) )]
        # remove the ceiling
        xyz = xyz[~( xyz[:,2]>(rm_y-thresh) )]
        # remove the wall
        xyz = xyz[~( xyz[:,1]>(rm_z-thresh) )]
        xyz = xyz[~( xyz[:,0]>(rm_x-thresh) )]
        xyz = xyz[~( xyz[:,0]<(-rm_x+thresh) )]

        # translate point clouds to the main camera view.
        T_w_main = main_cam_pose.inverse()
        xyz = (np.dot(T_w_main.rotation, xyz.T).T + 
            np.array([T_w_main.translation]))
        
    xyz = np.hstack( (xyz[:,[2]], -xyz[:,[0]], -xyz[:,[1]]) )

    return xyz

def generate_segmask_dataset(
    output_dataset_path, config
):
    """Generate a segmentation training dataset

    Parameters
    ----------
    dataset_path : str
        path to store the dataset
    config : dict (generate_mask_dataset.yaml)
        dictionary-like objects containing parameters of the simulator and visualization
    """

    # read subconfigs
    image_config = config["images"]

    # read camera instrinsics
    camera = config["state_space"]["camera"]
    camera_f = camera["focal_length"]
    camera_width = camera["im_width"]
    camera_height = camera["im_height"]

    # read room config
    room = config["state_space"]["heap"]["workspace"]
    room_halfWidth = float(room["width"])/2.0
    room_depth = float(room["length"])/2.0

    # debugging
    debug = config["debug"]
    if debug:
        np.random.seed(SEED)

    # read general parameters
    num_states = config["num_states"]                                  # 100
    num_images_per_state = config["num_images_per_state"]              # 5
    states_per_garbage_collect = config["states_per_garbage_collect"]  # 10

    # create the dataset path and all subfolders if they don't exist
    if not os.path.exists(output_dataset_path):
        os.mkdir(output_dataset_path)  # new_dataset
    color_dir = os.path.join(output_dataset_path, "color_ims")
    if image_config["color"] and not os.path.exists(color_dir):
        os.mkdir(color_dir)
    depth_dir = os.path.join(output_dataset_path, "depth_ims")
    if image_config["depth"] and not os.path.exists(depth_dir):
        os.mkdir(depth_dir)
    pcd_dir = os.path.join(output_dataset_path, "pcl_datas")
    if not os.path.exists(pcd_dir):
        os.mkdir(pcd_dir)

    # create the log file. remove the old one.
    experiment_log_filename = os.path.join(
        output_dataset_path, "operating_room.log"
    )
    if os.path.exists(experiment_log_filename):
        os.remove(experiment_log_filename)
    Logger.add_log_file(logger, experiment_log_filename, global_log_file=True)

    # Create initial env to generate metadata
    env = BinHeapEnv(config)

    # generate states and images
    state_id = 0
    while state_id < num_states:  # <100

        # sample states
        states_remaining = num_states - state_id
        # Number of states before garbage collection (due to pybullet memory issues)
        for i in range(min(states_per_garbage_collect, states_remaining)):  # (10, 100-state_id)

            # log current rollout
            if state_id % config["log_rate"] == 0:  # state_id%1, log every time.
                logger.info("State: %04d" % (state_id))

            try:
                # reset env
                env.reset()
                state = env.state

                # 同一场景生成多个相机视角，拼接成一片点云
                for k in range(num_images_per_state):  # 4

                    # reset the camera
                    cur_cam_pose = env.reset_camera()
                    # set w^T_cam and save b-box
                    if k == 0:
                        main_cam_pose = cur_cam_pose
                        T_tar_main = main_cam_pose.inverse().dot(state.cart_pose)
                        tar_rotation = np.arctan2(-T_tar_main.rotation[0,1],T_tar_main.rotation[2,1])
                        logger.info("b-box: [%.3f, %.3f, %.3f, %.3f]" % (
                            T_tar_main.translation[2], -T_tar_main.translation[0],
                            -T_tar_main.translation[1], tar_rotation))


                    obs = env.render_camera_image(color=image_config["color"])
                    if image_config["color"]:
                        color_obs, depth_obs = obs  # depth_obs: np.array(np.float32)
                    else:
                        depth_obs = obs

                    # from depth image to pcl data.
                    if k == 0:
                        points_data = depth2xyz(depth_obs, main_cam_pose, cur_cam_pose,
                            camera_f, camera_f, float(camera_width-1)/2.0,
                            float(camera_height-1)/2.0, num_images_per_state,
                            room_halfWidth, state.ceiling_height, room_depth)
                    else:    
                        points_data = np.vstack( (points_data,
                            depth2xyz(depth_obs, main_cam_pose, cur_cam_pose,
                                camera_f, camera_f, float(camera_width-1)/2.0,
                                float(camera_height-1)/2.0, num_images_per_state,
                                room_halfWidth, state.ceiling_height, room_depth)) )
                    

                    # Save depth image and semantic masks
                    if image_config["color"] and k==0:
                        ColorImage(color_obs).save(
                            os.path.join(
                                color_dir,
                                "image_{:04d}.png".format(
                                    state_id
                                )
                            )
                        )
                    if image_config["depth"] and k==0:
                        DepthImage(depth_obs).save(
                            os.path.join(
                                depth_dir,
                                "image_{:04d}.png".format(
                                    state_id
                                )
                            )
                        )
                
                # downsample and save point clouds.
                pcd_num = 40000
                points_data = points_data[np.random.choice(
                    points_data.shape[0], pcd_num, replace=False), :]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_data[:, :3])
                o3d.io.write_point_cloud(
                    os.path.join(
                        pcd_dir,
                        "pcd_{:04d}.ply".format(
                            state_id
                        )
                    ), pcd
                )

                # delete action objects
                for obj_state in state.obj_states:
                    del obj_state
                del state
                gc.collect()

                # update state id
                state_id += 1

            except Exception as e:
                # log an error
                logger.warning("Heap failed!")
                logger.warning("%s" % (str(e)))
                logger.warning(traceback.print_exc())
                if debug:
                    raise

                del env
                gc.collect()
                env = BinHeapEnv(config)

        # garbage collect
        del env
        gc.collect()
        env = BinHeapEnv(config)

    # logger.info(
    #     "Generated %d image datapoints" % (state_id * num_images_per_state)
    # )


if __name__ == "__main__":

    output_dataset_path = "output"
    config_filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),  # synthetic_data/
        "cfg/generate_mask_dataset.yaml"
    )

    # turn relative paths absolute
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(os.getcwd(), config_filename)

    # open config file
    config = YamlConfig(config_filename)

    # generate dataset
    generate_segmask_dataset(
        output_dataset_path,
        config
    )

