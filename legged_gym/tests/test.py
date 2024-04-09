import numpy as np
import matplotlib.pyplot as plt


def map_simulation_time_to_gait_time(simulation_time, gait_period):
    # 将仿真时间除以步态周期，得到周期性变化的因子
    factor = simulation_time / gait_period
    # 使用正弦函数将因子映射到[0, 2π]的正弦波形上
    sine_wave = np.sin(2 * np.pi * factor)
    # 将正弦波形映射到[0, 1]的范围，作为步态时间
    gait_time = (sine_wave + 1) / 2
    return gait_time


# 示例：将仿真时间从0秒到10秒映射为步态时间
simulation_time = np.linspace(0, 10, 1000)  # 生成仿真时间区间 [0, 10] 的100个点
gait_period = 1  # 步态周期为1秒
gait_time = map_simulation_time_to_gait_time(simulation_time, gait_period)
plt.figure()
plt.plot(gait_time)
plt.show()
# 打印部分结果
print(gait_time)
