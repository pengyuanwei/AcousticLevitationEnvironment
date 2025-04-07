import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from examples.utils.optimizer_utils_v2 import *

# 定义智能体类
class Agent:
    def __init__(self, start, goal, agent_id):
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.position = np.array(start, dtype=float)
        self.path = [self.position.copy()]
        self.id = agent_id


# 根据已规划智能体路径，获取在时刻 t 的位置
def get_dynamic_position(agent, t):
    # 如果规划步数不足，则取最后一个位置
    if t < len(agent.path):
        return agent.path[t]
    else:
        return agent.path[-1]
    

def get_dynamic_segment(agent, segment_length, t):
    '''
    获取已规划智能体的一个路径片段
    '''
    # 如果规划步数不足，则取最后一个位置
    if t < len(agent.path):
        return np.array(agent.path[t+1-segment_length:t+1])
    elif t >= len(agent.path)+segment_length-1:
        broadcasted_arr = np.broadcast_to(agent.path[-1], (segment_length, 3))
        return broadcasted_arr
    else:
        print('The last point:', agent.path[-1].shape)
        broadcasted_arr = np.broadcast_to(agent.path[-1], (t-len(agent.path), 3))
        temp_arr = np.array(agent.path[t-segment_length-len(agent.path):])
        return np.concatenate((temp_arr, broadcasted_arr))


# 吸引力函数，采用简单二次势函数
def compute_attractive_force(agent, k_att=1.0):
    # 吸引力方向指向目标
    return k_att * (agent.goal - agent.position)


# 排斥力函数，将其他智能体视为动态障碍物
def compute_repulsive_force(agent, dynamic_agents, d0=5.0, k_rep=100.0, t=0):
    force = np.zeros(3)
    for other in dynamic_agents:
        # 忽略自身
        if other.id == agent.id:
            continue
        # 获取其他智能体在时刻 t 的位置
        other_pos = get_dynamic_position(other, t)
        diff = agent.position - other_pos
        dist = np.linalg.norm(diff)
        # 当距离小于阈值且不为零时，产生排斥力
        if dist < d0 and dist > 1e-2:
            force += k_rep * (1.0/dist - 1.0/d0) / (dist**3) * diff
    return force


# 为单个智能体规划路径
def plan_path_for_agent(agent, dynamic_agents, k_att=1.0, k_rep=100.0, d0=5.0, dt=0.1, max_steps=1000, threshold=0.5):
    for t in range(max_steps):
        F_att = compute_attractive_force(agent, k_att)
        F_rep = compute_repulsive_force(agent, dynamic_agents, d0, k_rep, t)
        total_force = F_att + F_rep
        # 采用欧拉积分更新位置
        agent.position = agent.position + total_force * dt
        agent.path.append(agent.position.copy())
        if np.any(F_rep != 0.0) and t > 1:
            path_correction(agent, dynamic_agents, t)
        # 若距离目标足够近则终止规划
        if np.linalg.norm(agent.goal - agent.position) < threshold:
            break


def path_correction(agent, dynamic_agents, t):
    '''
    修正当前agent的之前的路径
    '''
    num_insert = 1
    segment = np.expand_dims(linear_interpolation(agent, num_insert, t), axis=1)
    print(segment.shape)
    other_segments = []
    for other in dynamic_agents:
        # 忽略自身
        if other.id == agent.id:
            continue
        # 获取其他智能体的segments
        segment_length = num_insert + 2
        other_segment = np.expand_dims(get_dynamic_segment(other, segment_length, t), axis=1)
        print('Other_segment:', other_segment.shape)
        other_segments.append(other_segment)
    other_segments = np.concatenate(other_segments, axis=1)
    print(other_segments.shape)
    all_segments = np.concatenate((segment, other_segments), axis=1)
    print(all_segments.shape)

    for i in range(all_segments.shape[0]-1):
        collision = collision_check(all_segments.shape[1], all_segments[i])
        if np.any(collision != 0):
            break
    if np.all(collision == 0):
        agent.path[t-1] = segment[1, 0, :]
    else:
        print('Collision!')


    print("AAA")


def linear_interpolation(agent, num_insert, t):
    # 取两个点
    p1 = np.array(agent.path[t-1-num_insert])  # 起点
    p2 = np.array(agent.path[t])    # 终点
    # 线性插值
    new_points = np.linspace(p1, p2, num=num_insert+2, endpoint=True)[1:]
    # 合并进原始 `segment`
    segment = np.vstack([p1, new_points])

    return segment


def main():
    # 示例：定义三个智能体
    agents = []
    agents.append(Agent(start=[0, 0, 0],   goal=[10, 10, 10], agent_id=1))
    agents.append(Agent(start=[10, 0, 0],  goal=[0, 10, 10],  agent_id=2))
    agents.append(Agent(start=[0, 10, 0],  goal=[10, 0, 10],  agent_id=3))
    
    planned_agents = []
    # 顺序为每个智能体规划路径
    for i, agent in enumerate(agents):
        # 对于未规划的智能体，这里只考虑已规划的作为动态障碍物
        dynamic_agents = planned_agents.copy()
        # 如果需要，也可以对未规划的智能体假设其保持在起点（或根据预估轨迹），这里简化处理
        plan_path_for_agent(agent, dynamic_agents, k_att=1.0, k_rep=100.0, d0=5.0, dt=0.15, max_steps=1000, threshold=0.5)
        planned_agents.append(agent)
        print()
    
    # 绘制三维路径
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for agent in agents:
        path = np.array(agent.path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], label=f"Agent {agent.id}")
        ax.scatter(agent.start[0], agent.start[1], agent.start[2], marker='o', s=50, c='green')
        ax.scatter(agent.goal[0], agent.goal[1], agent.goal[2], marker='*', s=100, c='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("Multi-Agent Path Planning using Artificial Potential Field")
    plt.show()


def collision_check(n_particles, coords):
    '''
    coords: (n_particles, 3)
    '''
    collision = np.zeros(n_particles)
    for i in range(n_particles):
        x, y, z = [coords[i][0], coords[i][1], coords[i][2]]            
        for j in range(i+1, n_particles):
            dist_square = (x - coords[j][0])**2 + (y - coords[j][1])**2 + (z - coords[j][2])**2
            if dist_square <= 25.0:
                collision[i] = 1.0
                collision[j] = 1.0

    return collision


if __name__ == "__main__":
    main()