import numpy as np
import matplotlib.pyplot as plt

file_name = r'/home/cvlab/additional-space/xj/data/nturgbd_raw/nturgb+d_skeletons/S001C001P001R001A018.skeleton'  # 13
max_V = 25  # 节点数
max_M = 2  # 骨架数量
with open(file_name, 'r') as fr:
    frame_num = int(fr.readline())
    point = np.zeros((3, frame_num, 25, 2))
    for frame in range(frame_num):
        person_num = int(fr.readline())
        for person in range(person_num):
            fr.readline()
            joint_num = int(fr.readline())
            for joint in range(joint_num):
                v = fr.readline().split(' ')
                if joint < max_V and person < max_M:
                    point[0, frame, joint, person] = float(v[0])  # 一个关节的一个坐标
                    point[1, frame, joint, person] = float(v[1])
                    point[2, frame, joint, person] = float(v[2])

print('read data done!')
print(point.shape)

# 通过最大最小的坐标值来选取合适的坐标轴
xmax = np.max(point[0, :, :, :]) + 0.5
xmin = np.min(point[0, :, :, :]) - 0.5
ymax = np.max(point[1, :, :, :]) + 0.5
ymin = np.min(point[1, :, :, :]) - 0.3
zmax = np.max(point[2, :, :, :])
zmin = np.min(point[2, :, :, :])

row = point.shape[1]  # 有多少帧

# 根据NTU骨架结构确定哪几个节点相连为骨骼
# 注意序号从0开始，需要减1
arms = np.array([24, 12, 11, 10, 9, 21, 5, 6, 7, 8, 20]) - 1  # 胳膊
rightHand = np.array([12, 25]) - 1  # 右手
leftHand = np.array([8, 23]) - 1  # 左手
legs = np.array([20, 19, 18, 17, 1, 13, 14, 15, 16]) - 1  # 腿
body = np.array([4, 3, 21, 2, 1]) - 1  # 身体

n = 0  # 从第n帧开始展示
m = 2  # 到第m帧结束，n<m<row，这个m可以选取一个小于最大帧数的阈值，便于查看，若m=1则展示一帧
plt.figure()
plt.ion()  # 使用plt.ion()这个函数，使matplotlib的显示模式转换为交互（interactive）模式。即使在脚本中遇到plt.show()，代码还是会继续执行。
color_point = 'black'  # 关节点颜色,可输入16进制调色板 #03ff
color_bone = 'black'  # 骨骼颜色  red
for i in range(n, m):
    plt.cla()  ## Clear axis即清除当前图形中的当前活动轴。其他轴不受影响。
    plt.scatter(point[0, i, :, :], point[1, i, :, :], c=color_point, s=40.0)  # 通过散点图绘制关节点
    # 通过直线图绘制两点间的连接线，即骨骼
    plt.plot(point[0, i, arms, 0], point[1, i, arms, 0], c=color_bone, lw=1.0)
    plt.plot(point[0, i, rightHand, 0], point[1, i, rightHand, 0], c=color_bone, lw=1.0)
    plt.plot(point[0, i, leftHand, 0], point[1, i, leftHand, 0], c=color_bone, lw=1.0)
    plt.plot(point[0, i, legs, 0], point[1, i, legs, 0], c=color_bone, lw=1.0)
    plt.plot(point[0, i, body, 0], point[1, i, body, 0], c=color_bone, lw=1.0)

    # 第二个骨架，如果有的话
    plt.plot(point[0, i, arms, 1], point[1, i, arms, 1], c=color_bone, lw=1.0)
    plt.plot(point[0, i, rightHand, 1], point[1, i, rightHand, 1], c=color_bone, lw=1.0)
    plt.plot(point[0, i, leftHand, 1], point[1, i, leftHand, 1], c=color_bone, lw=1.0)
    plt.plot(point[0, i, legs, 1], point[1, i, legs, 1], c=color_bone, lw=1.0)
    plt.plot(point[0, i, body, 1], point[1, i, body, 1], c=color_bone, lw=1.0)

    plt.axis('off')

    # plt.text(xmax - 0.5, ymax - 0.1, 'frame: {}/{}'.format(i, row - 1))  # 这是第几帧
    # plt.text(xmax-0.8, ymax-0.4, 'label: ' + str(label[i]))
    plt.xlim(xmin, xmax)  # 坐标轴
    plt.ylim(ymin, ymax)
    plt.savefig('../results/' + str(i) + '-skel.eps', dpi=150)
    # plt.pause(0.001)

plt.ioff()
# plt.tight_layout()
plt.show()