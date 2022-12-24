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

sys.path.append(os.getcwd())  # /home/neal/projects/synthetic_data, used for input syn module.
from syn.bin_heap_env import BinHeapEnv

SEED = 744

# set up logger
logger = Logger.get_logger("operating_room.py")

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

    # create the log file. remove the old one.
    experiment_log_filename = os.path.join(
        output_dataset_path, "operating_room.log"
    )
    if os.path.exists(experiment_log_filename):
        os.remove(experiment_log_filename)
    Logger.add_log_file(logger, experiment_log_filename, global_log_file=True)

    # Create initial env to generate metadata
    env = BinHeapEnv(config)
    # obj_id_map = env.state_space.obj_id_map  # stl 文件名 + id
    # obj_keys = env.state_space.obj_keys  # stl 文件名
    # mesh_filenames = env.state_space.mesh_filenames  # stl 文件地址

    # generate states and images
    state_id = 0
    while state_id < num_states:  # <100

        # sample states
        states_remaining = num_states - state_id
        # Number of states before garbage collection (due to pybullet memory issues)
        for i in range(min(states_per_garbage_collect, states_remaining)):  # (10, 100-state_id)

            # log current rollout
            if state_id % config["log_rate"] == 0:  # state_id%1, log every time.
                logger.info("State: %06d" % (state_id))

            try:
                # reset env
                env.reset()
                state = env.state

                # render images
                # 同一场景生成多个相机视角
                for k in range(num_images_per_state):  # 5

                    # reset the camera
                    if num_images_per_state > 1:
                        env.reset_camera()

                    obs = env.render_camera_image(color=image_config["color"])
                    if image_config["color"]:
                        color_obs, depth_obs = obs
                    else:
                        depth_obs = obs

                    # Save depth image and semantic masks
                    if image_config["color"]:
                        ColorImage(color_obs).save(
                            os.path.join(
                                color_dir,
                                "image_{:06d}.png".format(
                                    num_images_per_state * state_id + k
                                ),
                            )
                        )
                    if image_config["depth"]:
                        DepthImage(depth_obs).save(
                            os.path.join(
                                depth_dir,
                                "image_{:06d}.png".format(
                                    num_images_per_state * state_id + k
                                ),
                            )
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
                # env.state_space.obj_id_map = obj_id_map
                # env.state_space.obj_keys = obj_keys
                # env.state_space.mesh_filenames = mesh_filenames

        # garbage collect
        del env
        gc.collect()

    logger.info(
        "Generated %d image datapoints" % (state_id * num_images_per_state)
    )


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

