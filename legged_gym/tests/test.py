import torch
import matplotlib.pyplot as plt


def gait_test():
    # 2+2和 3+1步态相位曲线
    kappa = 0.05
    smoothing_cdf_start = torch.distributions.normal.Normal(0,
                                                            kappa).cdf

    t = torch.linspace(0, 1, 50)
    t1 = t + 0.5
    print(t)
    smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(t, 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(t, 1.0) - 0.5)) +
                               smoothing_cdf_start(torch.remainder(t, 1.0) - 1) * (
                                       1 - smoothing_cdf_start(
                                   torch.remainder(t, 1.0) - 0.5 - 1)))
    smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(t1, 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(t1, 1.0) - 0.5)) +
                               smoothing_cdf_start(torch.remainder(t1, 1.0) - 1) * (
                                       1 - smoothing_cdf_start(
                                   torch.remainder(t1, 1.0) - 0.5 - 1)))

    kappa = 0.03
    smoothing_cdf_start = torch.distributions.normal.Normal(0,
                                                            kappa).cdf
    t1 = t + 0.25
    t2 = t + 0.5
    t3 = t + 0.75
    smoothing_multiplier_FL_test = (smoothing_cdf_start(torch.remainder(t, 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(t, 1.0) - 0.75)) +
                                    smoothing_cdf_start(torch.remainder(t, 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(t, 1.0) - 0.75 - 1)))
    smoothing_multiplier_FR_test = (smoothing_cdf_start(torch.remainder(t1, 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(t1, 1.0) - 0.75)) +
                                    smoothing_cdf_start(torch.remainder(t1, 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(t1, 1.0) - 0.75 - 1)))
    smoothing_multiplier_HL_test = (smoothing_cdf_start(torch.remainder(t2, 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(t2, 1.0) - 0.75)) +
                                    smoothing_cdf_start(torch.remainder(t2, 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(t2, 1.0) - 0.75 - 1)))
    smoothing_multiplier_HR_test = (smoothing_cdf_start(torch.remainder(t3, 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(t3, 1.0) - 0.75)) +
                                    smoothing_cdf_start(torch.remainder(t3, 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(t3, 1.0) - 0.75 - 1)))
    # print(smoothing_multiplier_FL)
    # plt.plot(t.numpy(), smoothing_multiplier_FL.numpy())
    # plt.plot(t.numpy(), smoothing_multiplier_FR.numpy())
    plt.plot(t.numpy(), smoothing_multiplier_FL_test.numpy())
    plt.plot(t.numpy(), smoothing_multiplier_FR_test.numpy())
    plt.plot(t.numpy(), smoothing_multiplier_HL_test.numpy())
    plt.plot(t.numpy(), smoothing_multiplier_HR_test.numpy())


def gait_test2():
    # 模拟 _step_contact_targets中的相位变化
    frequencies = 1.0  # 步态频率
    phases = 0.5  # 相位offset0
    offsets = 0.25  # 相位offset1
    bounds = 0.0  # 相位offset1
    durations = 0.75  # 站立比例 [0~1]
    dt = 0.02
    end_time = 2

    gait_indices = torch.linspace(0, end_time, int(end_time / (dt * frequencies)))
    # gait_indices = torch.remainder(gait_indices, 1.0)
    foot_indices = [gait_indices + phases + offsets + bounds,
                    gait_indices + bounds,
                    gait_indices + offsets,
                    gait_indices + phases]
    foot_indices = torch.remainder(torch.cat([foot_indices[i] for i in range(4)]), 1.0).view(4, -1)
    # plt.plot(gait_indices.numpy(), gait_indices.numpy())

    # plt.plot(gait_indices.numpy(), foot_indices[0].numpy())
    # plt.plot(gait_indices.numpy(), foot_indices[1].numpy())
    # for idxs in foot_indices:
    #     stance_idxs = torch.remainder(idxs, 1) < durations
    #     swing_idxs = torch.remainder(idxs, 1) > durations
    #     idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations)
    #     idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations) * (
    #             0.5 / (1 - durations))
    kappa = 0.03
    smoothing_cdf_start = torch.distributions.normal.Normal(0,
                                                            kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2
    smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - durations)) +
                               smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                       1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - durations - 1)))
    smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - durations)) +
                               smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                       1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - durations - 1)))
    smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - durations)) +
                               smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                       1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - durations - 1)))
    smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - durations)) +
                               smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                       1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - durations - 1)))
    plt.plot(gait_indices.numpy(), smoothing_multiplier_FL.numpy())
    plt.plot(gait_indices.numpy(), smoothing_multiplier_FR.numpy())
    plt.plot(gait_indices.numpy(), smoothing_multiplier_RL.numpy())
    plt.plot(gait_indices.numpy(), smoothing_multiplier_RR.numpy())


if __name__ == '__main__':
    plt.figure(1)
    gait_test()
    plt.figure(2)
    gait_test2()
    plt.show()
