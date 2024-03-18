from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class BuptDogCfg(LeggedRobotCfg):
    class env:
        num_envs = 128
        num_observations = 235
        num_privileged_obs = None  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 12
        env_spacing = 3.0  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.393371]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "FL_X_joint": 0.0,  # [rad]
            "HL_X_joint": 0.0,  # [rad]
            "FR_X_joint": 0.0,  # [rad]
            "HR_X_joint": 0.0,  # [rad]
            "FL_Y_joint": 0.81521,  # [rad]
            "HL_Y_joint": 0.81521,  # [rad]
            "FR_Y_joint": 0.81521,  # [rad]
            "HR_Y_joint": 0.81521,  # [rad]
            "FL_KNEE_joint": -1.57079,  # [rad]
            "HL_KNEE_joint": -1.57079,  # [rad]
            "FR_KNEE_joint": -1.57079,  # [rad]
            "HR_KNEE_joint": -1.57079,  # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "P"
        stiffness = {"joint": 50.0}  # [N*m/rad]
        damping = {"joint": 2.0}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/buptDog/urdf/robot.urdf"
        name = "buptDog"
        foot_name = "FOOT"
        penalize_contacts_on = ["Y", "KNEE"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(LeggedRobotCfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0


class BuptDogCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rough_buptDog"
