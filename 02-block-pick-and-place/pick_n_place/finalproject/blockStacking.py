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
from robohive.utils.quat_math import euler2quat, quat2mat, mat2quat
from pick_n_place.utils.xml_utils import replace_simhive_path
from pathlib import Path
from time import sleep
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import click
import numpy as np
from robohive.utils.min_jerk import generate_joint_space_min_jerk
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RotationSpline
TARGET_BIN_POS = np.array([0.235, 0.5, 0.85])
BOX_BIN_POS = np.array([-0.235, 0.5, 0.85])
BIN_DIM = np.array([0.15, 0.25, 0])
BIN_TOP = 0.10
ARM_nJnt = 7

GRIPPER_LEFT_ACTUATOR = 7
GRIPPER_RIGHT_ACTUATOR = 8

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

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


import threading
from matplotlib.gridspec import GridSpec
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.prev_error = 0
        
    def set_target(self, target):
        self.target = target
        
    def update(self, current, dt):
        error = self.target - current
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative
    
    def reset(self):
        self.integral = 0
        self.prev_error = 0
        
    @property
    def error(self):
        return self.prev_error

class PIDArm:
    def __init__(self, sim: SimScene, kp, ki, kd, eff_name="end_effector"):
        self.sim = sim
        self.eff_name = eff_name
        self.pid_controllers = [PIDController(kp, ki, kd) for _ in range(ARM_nJnt)]
        self.current_values = [[] for _ in range(ARM_nJnt)]  # Store current values for plotting
        self.joint_min = sim.model.jnt_range[:, 0]
        self.joint_max = sim.model.jnt_range[:, 1]
        self.curr_eef_coords = []
        self.target_eef_coords = []
        self.pid_controller_targets = []
        self.plot_thread = threading.Thread(target=self.plot_current_values)
        self.plot_thread.start()  # Start the plotting thread

    def set_target(self, sim, targetPos, targetMat):
        # convert to joint space (solve IK)
        targetQuat = mat2quat(targetMat.reshape(3, 3))
        ik_result = qpos_from_site_pose(
            physics=sim,
            site_name=self.eff_name,
            target_pos=targetPos,
            target_quat=targetQuat,
            inplace=False,
            regularization_strength=1.0,
        )
        
        self.pid_controller_targets = []
        for i in range(ARM_nJnt):
            self.pid_controllers[i].set_target(ik_result.qpos[i])
            self.pid_controller_targets.append(ik_result.qpos[i])
        self.target_eef_coords = targetPos # only have x,y,z
        
    def update(self, sim, current_joint_positions, dt):  # returns control signals (joint torques)
        control_signals = [self.pid_controllers[i].update(current_joint_positions[i], dt) for i in range(ARM_nJnt)]
        sim.data.ctrl[:ARM_nJnt] = control_signals  # Update the control signals for the arm
        
        # Store current values for plotting
        for i in range(ARM_nJnt):
            self.current_values[i].append(current_joint_positions[i])
        self.curr_eef_coords = sim.data.site_xpos[sim.model.site_name2id(self.eff_name)].copy()
        return control_signals
    
    def error_to_target(self):
        return [self.pid_controllers[i].error for i in range(ARM_nJnt)]

    def plot_current_values(self):
        plt.ion()  # Turn on interactive mode
        def format_axes(fig):
            for i, ax in enumerate(fig.axes):
                ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
                ax.tick_params(labelbottom=False, labelleft=False)

        fig = plt.figure(figsize=(10, 8))
        gs = GridSpec(ARM_nJnt, 2, figure=fig)
        jointplots = []
        # add subplots for the joint plots
        for i in range(ARM_nJnt):
            jointplots.append(fig.add_subplot(gs[i, 0]))
        
        # add a big plot for the eef pos
        eefplot = fig.add_subplot(gs[0:, 1])
        # set xrange and yrange
        eefplot.set_xlim(-0.25, 0.25)
        eefplot.set_ylim(-0.5, 0.5)
        # NOTE: PLOTTING X AND Y IN REVERSE FOR EEF POS
        
        
        for i, ax in enumerate(jointplots):
            ax.set_ylim(self.joint_min[i], self.joint_max[i])
        while True:
            for i, ax in enumerate(jointplots):
                ax.cla()  # Clear the axis
                ax.plot(self.current_values[i], label='Current Value')  # Plot current values
                ax.axhline(y=self.pid_controller_targets[i], color='r', linestyle='--', label='Target Value')  # Target line
                ax.set_title(f'Joint {i} Current Value')
                ax.set_ylim(self.joint_min[i], self.joint_max[i])
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Current Value')
            
            # plot eef pos on the right, with xy of eef and target as points on 2d
            eefplot.cla()
            # only plot x,y as dots
            eefplot.plot(self.curr_eef_coords[1], self.curr_eef_coords[0], 'bo', label='Current Position')
            eefplot.plot(self.target_eef_coords[1], self.target_eef_coords[0], 'ro', label='Target Position')
            eefplot.set_title('End Effector and Target Position')
            eefplot.set_xlim(-1, 1)
            eefplot.set_ylim(-1, 1)
            eefplot.set_xlabel('Y Position')
            eefplot.set_ylabel('X Position')
            eefplot.legend()
            
            plt.pause(0.1)  # Pause to update the plot
            
def move_arm_PID(arm: PIDArm, sim: SimScene, target_pos, target_mat, time=5, hz=500):
    arm.set_target(sim, target_pos, target_mat)
    for _ in range(time * hz):
        current_joint_positions = sim.data.qpos[:ARM_nJnt]  # Get current joint positions
        dt = sim.model.opt.timestep  # Get the time step
        arm.update(sim, current_joint_positions, dt)  # Update the PID controller
        sim.advance(substeps=1, render=True)  # Advance the simulation
        
def move_arm_spline_PID(arm: PIDArm, sim: SimScene, start_pos, start_mat, end_pos, end_mat, 
                        waypoints=None, tracking_error=0.01, time=5, hz=500, num_intermediate_points=100):
    '''
    create a spline from start to end, with waypoints
    generate intermediate points from spline for PID control
    
    logic:
    - arm is at start
    - create array of 3d points to use as spline points
    - fit spline to points
    - generate intermediate points from spline
    - use PID to move to each intermediate point
        - if tracking error is less than threshold, set targets to next point's IK solution
    - repeat until end point is reached
    '''
    
    pointsArr = [start_pos]
    if waypoints is not None:
        pointsArr.extend(waypoints)
    pointsArr.append(end_pos)
    
    # fit spline to points
    spline = CubicSpline(np.linspace(0, 1, len(pointsArr)), pointsArr)
    # generate intermediate points from spline
    intermediate_times = np.linspace(0, time, num_intermediate_points)
    intermediate_points = spline(np.linspace(0, 1, num_intermediate_points))
    
    # convert start and end rotation matrices to quaternions, and then to Rotation objects
    start_quat = mat2quat(start_mat.reshape(3, 3))
    end_quat = mat2quat(end_mat.reshape(3, 3))
    rot = R.from_quat([start_quat, end_quat])
    
    # now use the rotation spline to interpolate between start and end rotations
    rotation_spline = RotationSpline(intermediate_times, rot)




def calculate_pos_above(sim, obj_name, z_offset=0.1): # returns pos matrix and rot matrix
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
    
    init_targets(sim, target_sid, box_sid)

    pid_arm = PIDArm(sim, kp=10.0, ki=0.0, kd=0.0)
    # every second, print error to target
    prev_time = 0

    while True:
        # current_joint_positions = sim.data.qpos[:ARM_nJnt]  # Get current joint positions
        # dt = sim.model.opt.timestep  # Get the time step
        # pid_arm.update(sim, current_joint_positions, dt)  # Update the PID controller
        # sim.advance(substeps=1, render=True)  # Advance the simulation
        # if sim.data.time - prev_time > 1:
        #     print("Error to target:", pid_arm.error_to_target())
        #     prev_time = sim.data.time
        
        init_targets(sim, target_sid, box_sid)
        open_gripper(sim)
        
        # Calculate position above the box
        pos_above, mat_above = calculate_pos_above(sim, "box", 0.02)
        # Move arm to the position above the box using PID control
        print("Moving arm to the position above the box")
        move_arm_PID(pid_arm, sim, pos_above, mat_above, time=3, hz=500)

        # now move to the box
        print("Moving arm to the box")
        box_sid, box_pos, box_mat = get_id_pos_rot(sim, "box")
        move_arm_PID(pid_arm, sim, box_pos, box_mat, time=3, hz=500)
        
        # close the gripper
        print("Closing the gripper")
        close_gripper(sim)

        # Move arm to pick target using PID control
        _, pick_target_pos, pick_target_mat = get_id_pos_rot(sim, "pick_target")
        print("Moving arm to pick target")
        move_arm_PID(pid_arm, sim, pick_target_pos, pick_target_mat, time=3, hz=500)

        # move to drop target
        pos_above, mat_above = calculate_pos_above(sim, "drop_target", 0.1)
        print("Moving arm to drop target")
        move_arm_PID(pid_arm, sim, pos_above, mat_above, time=2, hz=500)

        # open the gripper
        print("Opening the gripper")
        open_gripper(sim)

        sim.reset()

if __name__ == "__main__":
    main()
