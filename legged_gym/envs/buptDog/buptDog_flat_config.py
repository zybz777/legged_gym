from legged_gym.envs import BuptDogCfg, BuptDogCfgPPO


class BuptDogFlatCfg(BuptDogCfg):
    class env(BuptDogCfg.env):
        num_observations = 48  # 48
        num_envs = 4096

    class terrain(BuptDogCfg.terrain):
        mesh_type = "plane"
        measure_heights = False

    class asset(BuptDogCfg.asset):
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(BuptDogCfg.rewards):
        max_contact_force = 120.0

        class scales(BuptDogCfg.rewards.scales):
            tracking_lin_vel = 1.5
            lin_vel_z = -2.5
            orientation = -5.0
            feet_air_time = 2.0

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
        max_iterations = 1500
