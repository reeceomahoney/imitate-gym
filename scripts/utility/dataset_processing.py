import numpy as np
import pandas as pd
import os
from scipy.spatial.transform import Rotation as R
import pinocchio as pin


def get_foot_positions(joint_pos, model, data, foot_frames):
    foot_positions = np.zeros(12)
    pin.forwardKinematics(model, data, joint_pos)

    for i, frame in enumerate(foot_frames):
        frame_id = model.getFrameId(frame)
        pin.updateFramePlacement(model, data, frame_id)
        foot_positions[3*i:3*(i+1)] = data.oMf[frame_id].translation

    return foot_positions


# expert dataset
current_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
states = pd.read_csv(current_dir + "/../../resources/expert_data/expert_data.csv")

# reduce recorded frequency to 25Hz (400/25=16)
states = states.iloc[::16, :]
n_transitions = len(states.index)

# ------------------------------
# build column list
# ------------------------------

# orientation
orientation_list = ["field.pose.pose.orientation.x",
                    "field.pose.pose.orientation.y",
                    "field.pose.pose.orientation.z",
                    "field.pose.pose.orientation.w"]

# linear and angular com velocity
lin_vel_list = ["field.twist.twist.linear.x",
                "field.twist.twist.linear.y",
                "field.twist.twist.linear.z"]
ang_vel_list = ["field.twist.twist.angular.x",
                "field.twist.twist.angular.y",
                "field.twist.twist.angular.z"]

# joint positions and velocity
joint_pos_list = ["field.joints.position" + str(i) for i in range(12)]
joint_vel_list = ["field.joints.velocity" + str(i) for i in range(12)]

# ------------------------------
# build observations
# ------------------------------

# 33 cols for observation information plus 5 extra for random initialisation
obs = np.zeros((n_transitions, 50))

# rotation matrices
rot_mat_arr = [R.from_quat([row[0], row[1], row[2], row[3]]).as_matrix() for row in states[orientation_list].to_numpy()]

# gravity axis
obs[:, :3] = [rot[2].T for rot in rot_mat_arr]

# joint angles
obs[:, 3:15] = states[joint_pos_list].to_numpy()

# angular velocity
obs[:, 15:18] = [np.dot(row[0].T, row[1]) for row in zip(rot_mat_arr, states[ang_vel_list].to_numpy())]

# joint velocities
obs[:, 18:30] = states[joint_vel_list].to_numpy()

# based linear velocity
obs[:, 30:33] = [np.dot(row[0].T, row[1]) for row in zip(rot_mat_arr, states[lin_vel_list].to_numpy())]

# foot positions
# anymal_model = pin.buildModelFromUrdf(current_dir + "/../../rsc/anymal/urdf/anymal.urdf")
# anymal_data = anymal_model.createData()
# frame_list = ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
#
# obs[:, 33:45] = [get_foot_positions(row, anymal_model, anymal_data, frame_list) for row in
#                  states[joint_pos_list].to_numpy()]

# append extra information for random initialisation
obs[:, -5] = states['field.pose.pose.position.z']
obs[:, -4:] = states[orientation_list]

# export to csv
np.savetxt("../../resources/expert_data/expert_data_processed.csv", obs, delimiter=",")
