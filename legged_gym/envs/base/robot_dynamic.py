import multiprocessing
from functools import partial
import numpy as np
import pinocchio as pin
from numba import njit

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg


class RobotDynamic:
    def __init__(self, cfg: LeggedRobotCfg, num_envs):
        self.num_envs = num_envs
        self.cfg = cfg
        urdf_file = self.cfg.asset.file
        self.asset_path = urdf_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        self.root_joint = pin.JointModelFreeFlyer()
        self.robot_model = pin.buildModelFromUrdf(self.asset_path, self.root_joint)  # 浮动基模型
        # self.robot_data = self.robot_model.createData()  # 机器人数据存储
        self.robot_data = [self.robot_model.createData() for _ in range(self.num_envs)]
        self.q = np.zeros((self.num_envs, self.robot_model.nq))
        self.dq = np.zeros((self.num_envs, self.robot_model.nv))
        # 足端
        self.foot_link = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
        self.foot_pos_in_world = np.zeros((self.num_envs, 3, 4))
        self.foot_pos_in_body = np.zeros((self.num_envs, 3, 4))
        self.foot_vel_in_body = np.zeros((self.num_envs, 3, 4))
        self.foots_jaco_in_world = np.zeros((self.num_envs, 12, 18))
        self.foot_jaco_in_body = np.zeros((self.num_envs, 4, 3, 3))
        self.pool = multiprocessing.Pool(processes=4)

    def step(self, q: np.ndarray, dq: np.ndarray):
        self.q = q
        self.dq = dq
        for i in range(self.num_envs):
            self.forward_kinematics(i)
        # print(self.foot_pos_in_world)
        # print("foot_pos", self.foot_pos_in_body)

    def forward_kinematics(self, id):
        pin.forwardKinematics(self.robot_model, self.robot_data[id], self.q[id])
        pin.updateFramePlacements(self.robot_model, self.robot_data[id])
        base_pos = self.robot_data[id].oMf[self.robot_model.getFrameId("body")].translation[:, np.newaxis]  # 质心位置
        rot_mat = self.robot_data[id].oMf[self.robot_model.getFrameId("body")].rotation
        for i, link_name in enumerate(self.foot_link):
            self.foot_pos_in_world[id, :, i] = self.robot_data[id].oMf[
                self.robot_model.getFrameId(link_name)].translation
            J = pin.computeFrameJacobian(self.robot_model, self.robot_data[id], self.q[id],
                                         self.robot_model.getFrameId(link_name), pin.LOCAL_WORLD_ALIGNED)
            self.foots_jaco_in_world[id, 0 + 3 * i:3 + 3 * i, :] = J[:3, :]
            self.foot_jaco_in_body[id, i, :] = np.dot(np.transpose(rot_mat), J[:3, 6 + 3 * i:9 + 3 * i])
            self.foot_vel_in_body[id, :, i] = np.dot(self.foot_jaco_in_body[id, i, :], self.dq[id, 6 + 3 * i:9 + 3 * i])
        self.foot_pos_in_body[id] = np.dot(np.transpose(rot_mat), (self.foot_pos_in_world[id] - base_pos))


if __name__ == '__main__':
    robot = RobotDynamic(LeggedRobotCfg(), 2)
    q = np.zeros((robot.num_envs, 19))
    q[1][0] = 0.5
    dq = np.zeros((robot.num_envs, 18))
    robot.step(q, dq)
    # robot.forward_kinematics()
