from legged_gym.envs import BuptDogCfg, BuptDogCfgPPO


class BuptDogFlatCfg(BuptDogCfg):
    class env(BuptDogCfg.env):
        num_envs = 4096
        horizon = 20
        num_observations = 48 + 4 + 1
        # num_observations = (48 + 4 + 1 + 12) * horizon  # 48 + t_foot_phase + foot_height

    class terrain(BuptDogCfg.terrain):
        mesh_type = "plane"
        measure_heights = False

    class asset(BuptDogCfg.asset):
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(BuptDogCfg.rewards):
        max_contact_force = 100.0
        soft_dof_pos_limit = 0.8
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.31

        class scales(BuptDogCfg.rewards.scales):
            tracking_ang_vel = 1.0
            tracking_lin_vel = 4.0
            tracking_contacts_shaped_force = -0.0002
            tracking_contacts_shaped_vel = -0.0002
            raibert_heuristic = -0.001
            feet_clearance_cmd_linear = -0.01
            lin_vel_z = -1.0
            orientation = -2.0
            feet_air_time = 0.0

            termination = -0.0
            ang_vel_xy = -0.1
            torques = -2.5e-5
            dof_vel = -2.5e-5
            dof_acc = -2.5e-7
            base_height = -5.0
            collision = -1.0
            feet_stumble = -0.0
            action_rate = -2e-3
            action_rate2 = -2e-3
            stand_still = -0.00001

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
