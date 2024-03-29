import numpy as np
import pinocchio as pin
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.envs.buptDog.buptDog_flat_config import BuptDogFlatCfg
import torch


class RobotDynamic:
    def __init__(self, cfg: LeggedRobotCfg):
        self.cfg = cfg
        urdf_file = self.cfg.asset.file
        self.asset_path = urdf_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        self.root_joint = pin.JointModelFreeFlyer()
        self.robot_model = pin.buildModelFromUrdf(self.asset_path, self.root_joint)  # 浮动基模型
        self.robot_data = self.robot_model.createData()  # 机器人数据存储
        self.q_tensor = torch.zeros(self.robot_model.nq, dtype=torch.float, device='cuda')  # TODO: 目前只支持1个机器人,不是N个并行计算
        self.dq_tensor = torch.zeros(self.robot_model.nv, dtype=torch.float, device='cuda')
        self.q = self.q_tensor.cpu().numpy()
        self.dq = self.dq_tensor.cpu().numpy()
        # 足端
        self.foot_link = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
        self.foot_pos_in_world = np.zeros((3, 4))
        self.foot_pos_in_body = np.zeros((3, 4))
        # pin.forwardKinematics(self.robot_model, self.robot_data, self.q)
        # Print out the placement of each joint of the kinematic tree
        # for name, oMi in zip(self.robot_model.names, self.robot_data.oMi):
        #     print(("{:<24} : {: .2f} {: .2f} {: .2f}"
        #            .format(name, *oMi.translation.T.flat)))
        # Perform the forward kinematics over the kinematic tree

    def set_q(self, q: torch.tensor(19)):
        """
        :param q: 世界系基座位置+基座四元数(xyzw)+12个关节角度
        """
        self.q_tensor = q
        self.q = self.q_tensor.cpu().numpy()

    def set_dq(self, dq: torch.tensor(18)):
        """
        :param dq: 自身系基座速度+自身系基座角速度+12个关节角速度
        """
        self.dq = dq
        self.dq = self.dq_tensor.cpu().numpy()

    def step(self, q: torch.tensor(19), dq: torch.tensor(18)):
        self.set_q(q)
        self.set_dq(dq)
        self.forward_kinematics()

    def forward_kinematics(self):
        pin.forwardKinematics(self.robot_model, self.robot_data, self.q)
        pin.updateFramePlacements(self.robot_model, self.robot_data)
        base_pos = self.robot_data.oMf[self.robot_model.getFrameId("body")].translation  # 质心位置
        for i, link_name in enumerate(self.foot_link):
            self.foot_pos_in_world[:, i] = self.robot_data.oMf[self.robot_model.getFrameId(link_name)].translation
        # base_pos = self.robot_data.oMf
        print(self.foot_pos_in_world)


if __name__ == '__main__':
    robot = RobotDynamic(BuptDogFlatCfg())
    robot.forward_kinematics()
