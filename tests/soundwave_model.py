import torch
import math
import matplotlib.pyplot as plt
import numpy as np

class TransducerModel:
    def __init__(self, transducer):
        # transducer: Tensor of shape (n, 3)，假设只有一个换能器
        self.transducer = transducer

    def modal_model(self, points):
        """
        基于简化振动模式模型计算换能器阵列对目标点的贡献矩阵。
        假设换能器存在两个主要模态：刚性活塞模态和第一阶弯曲模态。

        参数：
          points: Tensor, 目标点坐标，shape (m,3)

        返回：
          trans_matrix: Tensor, 贡献矩阵，shape (m, n)
        """
        m = points.shape[0]
        n = self.transducer.shape[0]
        # 参数设置
        k = 2 * math.pi / 0.00865    # 波数，假设波长为0.00865m
        radius = 0.005               # 换能器半径，单位：m

        # 提取换能器和目标点坐标，分别调整形状为 (n,1) 和 (m,1)
        transducers_x = torch.reshape(self.transducer[:, 0], (n, 1))
        transducers_y = torch.reshape(self.transducer[:, 1], (n, 1))
        transducers_z = torch.reshape(self.transducer[:, 2], (n, 1))
        points_x = torch.reshape(points[:, 0], (m, 1))
        points_y = torch.reshape(points[:, 1], (m, 1))
        points_z = torch.reshape(points[:, 2], (m, 1))

        # 计算换能器到各目标点的距离（利用广播，得到 shape 为 (m, n)）
        distance = torch.sqrt((points_x - transducers_x.T)**2 +
                              (points_y - transducers_y.T)**2 +
                              (points_z - transducers_z.T)**2)
        # 计算平面距离（用于确定角度），shape 同 distance
        planar_distance = torch.sqrt((points_x - transducers_x.T)**2 +
                                     (points_y - transducers_y.T)**2)
        # 定义一个归一化参数，类似于bessel函数自变量
        bessel_arg = k * radius * torch.divide(planar_distance, distance)

        # # 计算目标点相对于换能器的发射角theta
        # theta = torch.atan2(planar_distance, distance)

        # 模态0：活塞模态（采用原有的多项式近似）
        directivity0 = (1/2 - bessel_arg**2 / 16 +
                        bessel_arg**4 / 384 - bessel_arg**6 / 18432 +
                        bessel_arg**8 / 1474560 - bessel_arg**10 / 176947200)

        # 计算相位项
        phase = torch.exp(1j * k * distance)
        # 结合幅值因子（此处仍使用 2*8.02），得到贡献矩阵
        trans_matrix = 2 * 8.02 * torch.multiply(torch.divide(phase, distance), directivity0)

        return trans_matrix  # shape: (m, n)

# 定义换能器位置，假设单个换能器位于原点 (0,0,0)，shape (1,3)
transducer_pos = torch.tensor([[0.0, 0.0, 0.0]])
model = TransducerModel(transducer_pos)

# 生成极坐标图数据：在半径为1m处，角度范围从 -90° 到 90°（即 -pi/2 到 pi/2）
num_points = 180
angles = np.linspace(-np.pi/2, np.pi/2, num_points)
r_fixed = 1.0  # 1 meter 距离

# 将极坐标转换为笛卡尔坐标（假设在垂直平面上，即 y=0）
x_points = r_fixed * np.sin(angles)
y_points = np.zeros_like(x_points)  # y=0
z_points = r_fixed * np.cos(angles)

# 构建目标点数组，shape (num_points, 3)
points = torch.tensor(np.stack([x_points, y_points, z_points], axis=-1), dtype=torch.float32)
print(points.shape)

# 计算 modal_model 输出，输出 shape 为 (m, n)，此处 m=180, n=1
trans_matrix = model.modal_model(points)
# 使用 squeeze() 将结果转换为一维数组，长度应为 180
amplitude = torch.abs(trans_matrix.squeeze()).detach().numpy()

# 将幅值转换为 dB（归一化到最大值）
amplitude_db = 20 * np.log10(amplitude / amplitude.max())
print(amplitude_db.shape)

# 绘制极坐标图
plt.figure(figsize=(8,6))
ax = plt.subplot(111, polar=True)
ax.plot(angles, amplitude_db)
# 设置角度范围为 0 到 270°，但自定义刻度标签显示为 -90° 到 90°
ax.set_thetamin(0)
ax.set_thetamax(270)
# 假设原始刻度为 [0, 45, 90, 135, 180, 225, 270]
ticks = np.array([0, 30, 60, 90, 270, 300, 330, 360])
# 将大于180的角度转换为负角度，例如 225-> -135, 270 -> -90
new_ticks = np.where(ticks > 180, ticks - 360, ticks)
ax.set_xticks(np.deg2rad(ticks))
ax.set_xticklabels(new_ticks)
ax.set_rlim(-50, 0)
ax.set_title("Radiation Directivity (Modal Model)", va='bottom')
ax.set_rlabel_position(90)
plt.show()
