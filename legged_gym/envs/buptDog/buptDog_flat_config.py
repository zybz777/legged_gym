from legged_gym.envs import BuptDogCfg, BuptDogCfgPPO


class BuptDogFlatCfg(BuptDogCfg):
    class env(BuptDogCfg.env):
        num_envs = 4096
        horizon = 30
        num_observations = (48 + 4 + 1 + 4)

    class terrain(BuptDogCfg.terrain):
        mesh_type = "plane"
        measure_heights = False

    class asset(BuptDogCfg.asset):
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(BuptDogCfg.rewards):
        # 支撑足
        max_contact_force = 100.0
        # 关节限幅
        soft_dof_pos_limit = 0.8
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        # 质心跟踪
        base_height_target = 0.31

        class scales(BuptDogCfg.rewards.scales):
            # 质心跟踪
            orientation = -5.0  # 角度跟踪
            tracking_ang_vel = 1.0  # Z轴角速度跟踪
            ang_vel_xy = -0.15  # xy角速度趋于0
            base_height = -5.0  # 质心高度跟踪
            tracking_lin_vel = 4.0  # xy线速度跟踪
            lin_vel_z = -2.0  # z轴线速度趋于0
            # 足端通用
            raibert_heuristic = -0.001  # 足端xy位置跟踪
            # 摆动腿
            tracking_contacts_shaped_force = -0.0004  # 摆动腿接触力趋于0
            feet_clearance_cmd_linear = -0.01  # 摆动腿Z轴高度跟踪
            feet_air_time = 1.0  # 摆动腿腾空时间大于0.5s
            feet_stumble = -0.0  # 摆动腿不要碰到垂直面
            # 支撑腿
            tracking_contacts_shaped_vel = -0.0002  # 支撑腿速度为0
            feet_slip = -0.2  # 支撑腿不滑动
            # 终止条件
            termination = -0.0
            # 碰撞惩罚
            collision = -2.0
            # 关节平滑
            dof_pos = -0.3  # 两帧之间位置减少突变
            dof_vel = -8e-5  # 两帧之间速度减少突变
            dof_acc = -5e-7  # 每帧加速度趋于0
            dof_acc_rate = 0  # 两帧之间加速度减少突变
            stand_still = -0.00001
            # 关节输出平滑
            action_rate = -2.5e-3  # 两帧之间输出位置减少突变
            action_rate2 = -2.5e-3  # 三帧之间输出位置减少突变
            torques = -3e-5  # 每帧输出力矩趋于0
            torques_rate = -3e-5  # 两帧之间输出力矩减少突变

    class commands(BuptDogCfg.commands):
        heading_command = False
        resampling_time = 4.0

        class ranges(BuptDogCfg.commands.ranges):
            ang_vel_yaw = [-1.5, 1.5]

    class domain_rand(BuptDogCfg.domain_rand):
        friction_range = [
            0.0,
            1.5,
        ]  # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.

    class noise:
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1


class BuptDogFlatCfgPPO(BuptDogCfgPPO):
    class policy(BuptDogCfgPPO.policy):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(BuptDogCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(BuptDogCfgPPO.runner):
        run_name = ""
        experiment_name = "flat_buptDog"
        load_run = -1
        max_iterations = 1000
