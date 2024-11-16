# """ =================================================
# Copyright (C) 2018 Vikash Kumar
# Adapted by Raghava Uppuluri for GE-AI Course
# Author  :: Vikash Kumar (vikashplus@gmail.com)
# Source  :: https://github.com/vikashplus/robohive
# License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
# ================================================= """

DESC = """
DESCRIPTION: Simple Box Pick and Place \n
HOW TO RUN:\n
    - Ensure poetry shell is activated \n
    - python3 main.py \n
"""

from robohive.physics.sim_scene import SimScene
from robohive.utils.inverse_kinematics import qpos_from_site_pose
from robohive.utils.min_jerk import *
from robohive.utils.quat_math import euler2quat, quat2mat, mat2quat
from pick_n_place.utils.xml_utils import replace_simhive_path

from dm_control.mujoco.wrapper.core import MjModel

# set env for gpu
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path
from time import sleep
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import click
import numpy as np

TARGET_BIN_POS = np.array([0.235, 0.5, 0.85])
BOX_BIN_POS = np.array([-0.235, 0.5, 0.85])
# BIN_DIM = np.array([0.2, 0.3, 0])
BIN_DIM = np.array([0.15, 0.25, 0])
BIN_TOP = 0.10
ARM_nJnt = 7

GRIPPER_LEFT_ACTUATOR = 7
GRIPPER_RIGHT_ACTUATOR = 8

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def _min_jerk_spaces(N: int, T: float):
    """
    Generates a 1-dim minimum jerk trajectory from 0 to 1 in N steps & T seconds.
    Assumes zero velocity & acceleration at start & goal.
    The resulting trajectories can be scaled for different start & goals.
    Args:
        N: Length of resulting trajectory in steps
        T: Duration of resulting trajectory in seconds
    Returns:
        p_traj: Position trajectory of shape (N,)
        pd_traj: Velocity trajectory of shape (N,)
        pdd_traj: Acceleration trajectory of shape (N,)
    """
    assert N > 1, "Number of planning steps must be larger than 1."

    t_traj = np.linspace(0, 1, N)
    p_traj = 10 * t_traj**3 - 15 * t_traj**4 + 6 * t_traj**5
    pd_traj = (30 * t_traj**2 - 60 * t_traj**3 + 30 * t_traj**4) / T
    pdd_traj = (60 * t_traj - 180 * t_traj**2 + 120 * t_traj**3) / (T**2)

    return p_traj, pd_traj, pdd_traj

def slerp_quaternion(q_start, q_end, t_array):
    """
    Performs spherical linear interpolation (SLERP) between two quaternions over a time array.

    Args:
        q_start: Starting quaternion (as a list or numpy array)
        q_end: Ending quaternion (as a list or numpy array)
        t_array: Array of time steps normalized between [0, 1]

    Returns:
        interpolated_quats: Interpolated quaternions for each time step
    """
    r_start = R.from_quat(q_start)
    r_end = R.from_quat(q_end)

    slerp = Slerp([0, 1], R.from_quat([q_start, q_end]))

    interpolated_rots = slerp(t_array)
    return interpolated_rots.as_quat()

def generate_waypoint_space_min_jerk(
    start_pos, goal_pos, start_quat, goal_quat, time_to_go: float, dt: float):
    """
    Generates a minimum jerk trajectory for both position and orientation (in quaternions).

    Args:
        start_pos: Start position (3D) as a numpy array
        goal_pos: Goal position (3D) as a numpy array
        start_quat: Start orientation as a quaternion (4D) numpy array
        goal_quat: Goal orientation as a quaternion (4D) numpy array
        time_to_go: Duration of the trajectory
        dt: Time step duration

    Returns:
        waypoints: List of waypoints with position, velocity, acceleration, and orientation (quaternion).
    """
    steps = int(time_to_go / dt)

    # Generate position trajectory using minimum jerk
    p_traj, pd_traj, pdd_traj = _min_jerk_spaces(steps, time_to_go)

    # Linear interpolation for position (scaling by trajectory)
    D_pos = goal_pos - start_pos
    pos_traj = start_pos[None, :] + D_pos[None, :] * p_traj[:, None]
    vel_traj = D_pos[None, :] * pd_traj[:, None]
    accel_traj = D_pos[None, :] * pdd_traj[:, None]

    # Interpolate quaternions (SLERP for smooth orientation changes)
    t_traj = np.linspace(0, 1, steps)
    quat_traj = slerp_quaternion(start_quat, goal_quat, t_traj)

    waypoints = [
        {
            "time_from_start": i * dt,
            "position": pos_traj[i, :],
            "velocity": vel_traj[i, :],
            "acceleration": accel_traj[i, :],
            "orientation": quat_traj[i, :],  # Quaternion at each time step
        }
        for i in range(steps)
    ]

    return waypoints



# Function to open the gripper
def open_gripper(sim):
    sim.data.ctrl[GRIPPER_LEFT_ACTUATOR] = 0.04
    sim.data.ctrl[GRIPPER_RIGHT_ACTUATOR] = 0.04

# Function to close the gripper
def close_gripper(sim):
    sim.data.ctrl[GRIPPER_LEFT_ACTUATOR] = 0.0
    sim.data.ctrl[GRIPPER_RIGHT_ACTUATOR] = 0.0

def is_gripper_open(sim):
    return sim.data.ctrl[GRIPPER_LEFT_ACTUATOR] == 0.04 and sim.data.ctrl[GRIPPER_RIGHT_ACTUATOR] == 0.04

def init_targets(sim: SimScene, target_id, box_id):

    # # first print all objects and ids currently in the sim
    # model: MjModel = sim.model

    # for i in range(sim.model.njnt):
    #     print("joint_id:", i, "joint_name:", sim.model.id2name(i, "joint"))

    # Sample random place targets
    target_pos = (
        TARGET_BIN_POS
        + np.random.uniform(high=BIN_DIM, low=-1 * BIN_DIM)
        + np.array([0, 0, BIN_TOP])
    )  # add some z offset
    target_elr = np.random.uniform(high=[3.14, 0, 0], low=[3.14, 0, -3.14])
    target_quat = euler2quat(target_elr)
    # propagate targets to the sim for viz (ONLY FOR VISUALIZATION)
    sim.model.site_pos[target_id][:] = target_pos - np.array([0, 0, BIN_TOP])
    sim.model.site_quat[target_id][:] = target_quat

    # Sample random box position and orientation
    box_pos = (
        BOX_BIN_POS
        + np.random.uniform(high=BIN_DIM, low=-1 * BIN_DIM)
        + np.array([0, 0, BIN_TOP])
    )  # add some z offset
    box_elr = np.random.uniform(high=[3.14, 0, 0], low=[3.14, 0, -3.14])
    box_quat = euler2quat(box_elr)

    # Get the joint ID for the box's freejoint
    box_joint_id = sim.model.joint_name2id('box_freejoint')

    # Calculate the starting index of the joint's qpos based on the joint ID
    qpos_start = sim.model.jnt_qposadr[box_joint_id]

    # Update the position and orientation of the box via its free joint
    sim.data.qpos[qpos_start : qpos_start + 3] = box_pos  # position (x, y, z)
    sim.data.qpos[qpos_start + 3 : qpos_start + 7] = box_quat  # quaternion (w, x, y, z)

    # reseed the arm for IK (ONLY FOR VISUALIZATION)
    ARM_JNT0 = np.mean(sim.model.jnt_range[:ARM_nJnt], axis=-1)
    sim.data.qpos[:ARM_nJnt] = ARM_JNT0
    sim.forward()

# def plotTrajectory(start, goal, eefpos, eefrot):
#     print("Start Position:", start)
#     print("Goal Position:", goal)
#     print("Start End-effector Position:", eefpos[0])
#     print("End End-effector Position:", eefpos[-1])
#     print("Start End-effector Orientation:\n", eefrot[0])
#     print("End End-effector Orientation:\n", eefrot[-1])

#     eefpos = np.array(eefpos)  # Convert list to numpy array

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Extract x, y, z coordinates
#     x = eefpos[:, 0]
#     y = eefpos[:, 1]
#     z = eefpos[:, 2]

#     # Plot the start and end points
#     ax.scatter(start[0], start[1], start[2], c='r', marker='o', label="Start")
#     ax.scatter(goal[0], goal[1], goal[2], c='g', marker='o', label="Goal")

#     # Plot the trajectory
#     ax.plot(x, y, z, label="End-effector Trajectory", color="b")

#     # Now plot the orientation vectors
#     for i in range(0, len(eefpos), 100):  # Plot orientation every 10 steps for clarity
#         pos = eefpos[i]
#         rot_mat = eefrot[i]

#         # Extract the x, y, and z axes of the rotation matrix
#         x_axis = rot_mat[:, 0]  # X axis of the end-effector
#         y_axis = rot_mat[:, 1]  # Y axis of the end-effector
#         z_axis = rot_mat[:, 2]  # Z axis of the end-effector

#         # Plot the x-axis as a line originating from the end-effector position
#         ax.quiver(pos[0], pos[1], pos[2], x_axis[0], x_axis[1], x_axis[2],
#                   length=0.1, color='r', label="X-axis" if i == 0 else "")
#         # Plot the y-axis as a line originating from the end-effector position
#         ax.quiver(pos[0], pos[1], pos[2], y_axis[0], y_axis[1], y_axis[2],
#                   length=0.1, color='g', label="Y-axis" if i == 0 else "")
#         # Plot the z-axis as a line originating from the end-effector position
#         ax.quiver(pos[0], pos[1], pos[2], z_axis[0], z_axis[1], z_axis[2],
#                   length=0.1, color='b', label="Z-axis" if i == 0 else "")



def plotTrajectory(start, goal, eefpos, eefrot):
    print("Start Position:", start)
    print("Goal Position:", goal)
    print("Start End-effector Position:", eefpos[0])
    print("End End-effector Position:", eefpos[-1])
    print("Start End-effector Orientation:\n", eefrot[0])
    print("End End-effector Orientation:\n", eefrot[-1])
    
    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot start and goal points
    ax.scatter(start[0], start[1], start[2], c='green', marker='o', s=100, label='Start')
    ax.scatter(goal[0], goal[1], goal[2], c='red', marker='x', s=100, label='Goal')

    # Convert eefpos to numpy array for easier manipulation
    eefpos = np.array(eefpos)

    # Plot the trajectory (3D line)
    ax.plot(eefpos[:, 0], eefpos[:, 1], eefpos[:, 2], label='Trajectory', color='blue')

    # Plot the rotation at each point
    for i, (pos, rot) in enumerate(zip(eefpos, eefrot)):
        if i % 100 == 0:
            # Get the origin of the rotation and the axes vectors (columns of the rotation matrix)
            origin = pos
            x_axis = rot[:, 0]  # x-axis direction
            y_axis = rot[:, 1]  # y-axis direction
            z_axis = rot[:, 2]  # z-axis direction

            # Scale the axes for visualization
            scale = 0.05
            ax.quiver(origin[0], origin[1], origin[2],
                      x_axis[0], x_axis[1], x_axis[2], color='r', length=scale, normalize=True, label='X-axis')
            ax.quiver(origin[0], origin[1], origin[2],
                      y_axis[0], y_axis[1], y_axis[2], color='g', length=scale, normalize=True, label='Y-axis')
            ax.quiver(origin[0], origin[1], origin[2],
                      z_axis[0], z_axis[1], z_axis[2], color='b', length=scale, normalize=True, label='Z-axis')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Set the range of the axes
    # ax.set_xlim([-0.5, 0.5])
    # ax.set_ylim([0.35, 0.75])
    # ax.set_zlim([0.7, 1.7])
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 2])


    # set size of the axes (2:1:1 ratio)
    ax.set_box_aspect([1, 1, 1])

    # set camera angle (default looks from +x, -y, +z), change to look from -x, -y, +z
    ax.view_init(elev=30, azim=-130)

    ax.set_title('End-effector Trajectory with Orientation (X, Y, Z axes)')

    # change plot size
    fig.set_size_inches(10, 8)
    plt.show()

def plot_rotmat(mat1, mat2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    mat1 = mat1.reshape(3, 3)
    mat2 = mat2.reshape(3, 3)

    # Extract the x, y, and z axes of the rotation matrix
    x_axis1 = mat1[:, 0]  # X axis of the end-effector
    y_axis1 = mat1[:, 1]  # Y axis of the end-effector
    z_axis1 = mat1[:, 2]  # Z axis of the end-effector

    x_axis2 = mat2[:, 0]  # X axis of the end-effector
    y_axis2 = mat2[:, 1]  # Y axis of the end-effector
    z_axis2 = mat2[:, 2]  # Z axis of the end-effector

    # Plot the x-axis as a line originating from the end-effector position
    ax.quiver(0, 0, 0, x_axis1[0], x_axis1[1], x_axis1[2], length=0.1, color='r', label="X-axis 1")
    ax.quiver(0, 0, 0, x_axis2[0], x_axis2[1], x_axis2[2], length=0.1, color='b', label="X-axis 2")

    # Plot the y-axis as a line originating from the end-effector position
    ax.quiver(0, 0, 0, y_axis1[0], y_axis1[1], y_axis1[2], length=0.1, color='g', label="Y-axis 1")
    ax.quiver(0, 0, 0, y_axis2[0], y_axis2[1], y_axis2[2], length=0.1, color='y', label="Y-axis 2")

    # Plot the z-axis as a line originating from the end-effector position
    ax.quiver(0, 0, 0, z_axis1[0], z_axis1[1], z_axis1[2], length=0.1, color='b', label="Z-axis 1")
    ax.quiver(0, 0, 0, z_axis2[0], z_axis2[1], z_axis2[2], length=0.1, color='r', label="Z-axis 2")

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.show()

def plot_quat(quat1, quat2):

    mat1 = quat2mat(quat1)
    mat2 = quat2mat(quat2)

    plot_rotmat(mat1, mat2)

def get_id_pos_rot(sim, name):
    # check if end_is is a site or body
    try:
        # get the site id
        target_id = sim.model.name2id(name, "site")
        target_pos = sim.data.site_xpos[sim.model.site_name2id(name)]
        target_mat = sim.data.site_xmat[sim.model.site_name2id(name)]
    except:
        target_id = sim.model.name2id(name, "body")
        target_pos = sim.data.xpos[sim.model.body_name2id(name)]
        target_mat = sim.data.xmat[sim.model.body_name2id(name)]

    return target_id, target_pos, target_mat

def move_arm(sim: SimScene, object_name, end_name, rotate=True, time=5, hz=60):

    _, target_pos, target_mat = get_id_pos_rot(sim, end_name)

    if end_name == "pick_target" or end_name == "drop_target":
        target_mat = -target_mat


    start_sid = sim.model.site_name2id(object_name)
    start_pos = sim.data.site_xpos[start_sid]
    start_mat = sim.data.site_xmat[start_sid]

    # check if the rotation is correct
    object_mat = sim.data.site_xmat[start_sid]

    # if the object matrix is off compared to the site matrix, correct the site matrix to match the object matrix
    if not rotate:
        target_mat = object_mat

    move_arm_with_pose(sim, start_pos, start_mat, target_pos, target_mat, time, hz)

def move_arm_with_pose(sim: SimScene, start_pos, start_mat, target_pos, target_mat=None, time=5, hz=60):
    if target_mat is not None:
        target_quat = mat2quat(target_mat.reshape(3, 3))
    else:
        target_quat = None

    if start_mat is None:
        # by default we are using end effector site
        # so get position and orientation of the end effector

        start_pos = sim.data.site_xpos[sim.model.site_name2id("end_effector")].copy()
        start_mat = sim.data.site_xmat[sim.model.site_name2id("end_effector")].copy()


    # plot the orientation of the start and end quaternions
    # if target_quat is not None:
    #     plot_rotmat(start_mat, target_mat)

    ik_result = qpos_from_site_pose(
        physics=sim,
        site_name="end_effector",
        target_pos=target_pos,
        target_quat=target_quat,
        inplace=False,
        regularization_strength=1.0,
    )

    print(
        "IK:: Success:{}, total steps:{}, err_norm:{}".format(
            ik_result.success, ik_result.steps, ik_result.err_norm
        )
    )

    if not ik_result.success:
        raise ValueError("IK failed, resampling target")

    ARM_JNTS = sim.data.qpos[:ARM_nJnt]

    print("target_pos:", target_pos)
    print("target_quat:", target_quat)
    print("arm initial pos:", ARM_JNTS)
    print("arm target pos:", ik_result.qpos[:ARM_nJnt])
    print("ttg:", time)
    print("sim.model.opt.timestep:", sim.model.opt.timestep)

    arm_waypoints = generate_joint_space_min_jerk(
        start=ARM_JNTS,
        goal=ik_result.qpos[:ARM_nJnt],
        time_to_go=time,
        dt=sim.model.opt.timestep,
    )

    print("arm_waypoints[:5]:", arm_waypoints[:5])

    eef_positions = []
    eef_orientations = []

    for i in range(len(arm_waypoints)):
        sim.data.ctrl[:ARM_nJnt] = arm_waypoints[i]["position"]
        sim.advance(render=True)

        eef_positions.append(sim.data.site_xpos[sim.model.site_name2id("end_effector")].copy())
        eef_orientations.append(sim.data.xmat[sim.model.site_name2id("end_effector")].copy().reshape(3, 3))

        # i want to print the orientation of the end effector on the sim
        # mj = sim.get_mjlib()
        # mj.mju_printMat(sim.data.xmat[sim.model.site_name2id("end_effector")].reshape(9), 3)

    plotTrajectory(start_pos, target_pos, eef_positions, eef_orientations)

def move_arm_WAYPOINT_SPACE(sim: SimScene, start_pos, start_mat, target_pos, target_mat=None, time=5, hz=500):
    if start_mat is None:
        # by default we are using end effector site
        # so get position and orientation of the end effector

        start_pos = sim.data.site_xpos[sim.model.site_name2id("end_effector")].copy()
        start_mat = sim.data.site_xmat[sim.model.site_name2id("end_effector")].copy()

    start_quat = mat2quat(start_mat.reshape(3, 3))

    if target_mat is not None:
        target_quat = mat2quat(target_mat.reshape(3, 3))
    else:
        target_quat = mat2quat(start_mat.reshape(3, 3))

    # using waypoint space
    waypoints = generate_waypoint_space_min_jerk(
        start_pos=start_pos,
        goal_pos=target_pos,
        start_quat=start_quat,
        goal_quat=target_quat,
        time_to_go=time,
        dt=sim.model.opt.timestep,
    )

    # now we have interpolated waypoints, need to IK all of them
    joint_positions = []
    for waypoint in waypoints:
        ik_result = qpos_from_site_pose(
            physics=sim,
            site_name="end_effector",
            target_pos=waypoint["position"],
            target_quat=waypoint["orientation"],
            inplace=False,
            regularization_strength=1.0,
        )

        print(
            "IK:: Success:{}, total steps:{}, err_norm:{}".format(
                ik_result.success, ik_result.steps, ik_result.err_norm
            )
        )

        if not ik_result.success:
            raise ValueError("IK failed, resampling target")

        joint_positions.append(ik_result.qpos[:ARM_nJnt])


    eef_positions = []
    eef_orientations = []

    # now we have joint positions over time, propagate them in the sim
    for i in range(len(joint_positions)):
        sim.data.ctrl[:ARM_nJnt] = joint_positions[i]
        sim.advance(render=True)

        eef_positions.append(sim.data.site_xpos[sim.model.site_name2id("end_effector")].copy())
        eef_orientations.append(sim.data.xmat[sim.model.site_name2id("end_effector")].copy())

    # plotTrajectory(start_pos, target_pos, eef_positions, eef_orientations)


def calculate_pos_above(sim, obj_name, z_offset=0.1):
    # Get the object's position
    obj_id, obj_pos, obj_mat = get_id_pos_rot(sim, obj_name)

    # Get the object's dimensions
    obj_dim = sim.model.geom_size[sim.model.body_geomadr[obj_id]]

    # Calculate the position above the object
    pos_above = obj_pos + np.array([0, 0, obj_dim[2] + z_offset])
    mat_above = obj_mat

    return pos_above, mat_above

@click.command(help=DESC)
@click.option(
    "-s",
    "--sim_path",
    type=str,
    help="environment to load",
    required=True,
    default="pick_place.xml",
)

@click.option("-h", "--horizon", type=int, help="time (s) to simulate", default=2)
def main(sim_path, horizon):
    # Prep
    sim_path = Path(__file__).parent.parent.absolute() / "env" / sim_path
    sim_xml = replace_simhive_path(str(sim_path))
    print(f"Loading {sim_xml}")
    sim: SimScene = SimScene.get_sim(model_handle=sim_xml)

    # # setup
    target_sid = sim.model.site_name2id("drop_target")
    box_sid = sim.model.body_name2id("box")
    # pick_sid = sim.model.site_name2id("pick_target")
    # eef_sid = sim.model.site_name2id("end_effector")

    # while True:
    #     try:
    #         # Update targets
    #         init_targets(sim, target_sid, box_sid)
    #         open_gripper(sim)
    #         site_name = "box"
    #         pos_above, mat_above = calculate_pos_above(sim, site_name, 0.02)
    #         move_arm_with_pose(sim, None, None, pos_above, mat_above, time=3, hz=60)

    #         move_arm(sim, "end_effector", "box", rotate=True, time=2, hz=200)

    #         close_gripper(sim)
    #         # pos_above, mat_above = calculate_pos_above(sim, "box", 0.2)
    #         move_arm_with_pose(sim, None, None, pos_above, mat_above, time=2, hz=60)

    #         move_arm(sim, "end_effector", "pick_target", rotate=True, time=3, hz=60)
    #         pos_above, mat_above = calculate_pos_above(sim, "drop_target", 0.05)
    #         move_arm_with_pose(sim, None, None, pos_above, mat_above, time=3, hz=60)

    #         open_gripper(sim)
    #         pos_above, mat_above = calculate_pos_above(sim, "drop_target", 0.1)
    #         move_arm_with_pose(sim, None, None, pos_above, mat_above, time=2, hz=60)
    #         move_arm(sim, "end_effector", "pick_target", rotate=True, time=3, hz=60)

    #         sim.reset()
    #     except ValueError as e:
    #         print(e)
    #         print("IK faiiled")
    #         sim.reset()

    while True:
        try:
            # Update targets
            init_targets(sim, target_sid, box_sid)
            open_gripper(sim)
            site_name = "box"
            pos_above, mat_above = calculate_pos_above(sim, site_name, 0.02)
            move_arm_WAYPOINT_SPACE(sim, None, None, pos_above, mat_above, time=3, hz=60)

            move_arm(sim, "end_effector", "box", rotate=True, time=2, hz=200)

            close_gripper(sim)
            # pos_above, mat_above = calculate_pos_above(sim, "box", 0.2)
            move_arm_WAYPOINT_SPACE(sim, None, None, pos_above, mat_above, time=2, hz=60)

            move_arm(sim, "end_effector", "pick_target", rotate=True, time=3, hz=60)
            pos_above, mat_above = calculate_pos_above(sim, "drop_target", 0.05)
            move_arm_WAYPOINT_SPACE(sim, None, None, pos_above, mat_above, time=3, hz=60)

            open_gripper(sim)
            pos_above, mat_above = calculate_pos_above(sim, "drop_target", 0.1)
            move_arm_WAYPOINT_SPACE(sim, None, None, pos_above, mat_above, time=2, hz=60)
            move_arm(sim, "end_effector", "pick_target", rotate=True, time=3, hz=60)

            sim.reset()
        except ValueError as e:
            print(e)
            print("IK faiiled")
            sim.reset()

if __name__ == "__main__":
    main()
