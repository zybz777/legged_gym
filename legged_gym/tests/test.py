import torch
import matplotlib.pyplot as plt

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
print(smoothing_multiplier_FL)
plt.plot(t.numpy(), smoothing_multiplier_FL.numpy())
plt.plot(t.numpy(), smoothing_multiplier_FR.numpy())
plt.show()