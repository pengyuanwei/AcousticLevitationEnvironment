import os
import math
import gymnasium as gym
from acoustorl import MADDPG

from examples.utils.general_utils import *
from examples.utils.top_bottom_setup import top_bottom_setup

'''
Change from generate_path_2.py: path finding with gorkov correction
需要提前确定的参数：
    - 时间步长 delta t
    - Gorkov optimization:
        - Candidate solution 的数量
        - Search domain 的半径
    - Path smoothing:
        - Keypoints 之间插入的点的数量
'''

if __name__ == "__main__":
    n_particles = 8
    global_model_dir_1 = './experiments/experiment_20'
    global_model_dir_2 = './experiments/experiment_19'
    global_model_dir_3 = './experiments/experiment_98'
    global_model_dir_4 = './experiments/experiment_99'
    best_model_number_1 = 1000
    best_model_number_2 = 1000

    save_dir = os.path.join(global_model_dir_1, '20_19_98_99')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # WGS, iterations=5
    levitator = top_bottom_setup(n_particles, algorithm='Naive', iterations=5)

    env_name_1 = "acousticlevitationenvironment/PlannerAPFGorkov-v0"
    delta_time_1 = 0.096
    max_timesteps_1 = 20
    global_env = gym.make(env_name_1, n_particles=n_particles, delta_time=delta_time_1, max_timesteps=max_timesteps_1, levitator=levitator)

    env_name_2 = "acousticlevitationenvironment/RePlannerAPF-v0"
    delta_time_2 = 0.096
    max_timesteps_2 = 20
    replan_env = gym.make(env_name_2, n_particles=n_particles, delta_time=delta_time_2, max_timesteps=max_timesteps_2)

    state_dim = global_env.observation_space[0].shape[0]
    action_dim = global_env.action_space[0].shape[0]
    min_action = global_env.action_space[0].low[0]
    max_action = global_env.action_space[0].high[0]  # 动作最大值

    hidden_dim = 64

    main_agent = MADDPG(
        num_agents = n_particles, 
        state_dims = state_dim, 
        action_dims = action_dim, 
        min_action = min_action,
        max_action = max_action,         
        critic_input_dim = 9*n_particles, 
        hidden_dim = hidden_dim,
    )
    main_agent.load(best_model_number_1, global_model_dir_1)

    replan_agent = MADDPG(
        num_agents = n_particles, 
        state_dims = state_dim, 
        action_dims = action_dim, 
        min_action = min_action,
        max_action = max_action,         
        critic_input_dim = 9*n_particles, 
        hidden_dim = hidden_dim,
    )
    replan_agent.load(best_model_number_2, global_model_dir_2)

    replan_agent_1 = MADDPG(
        num_agents = n_particles, 
        state_dims = state_dim, 
        action_dims = action_dim, 
        min_action = min_action,
        max_action = max_action,         
        critic_input_dim = 9*n_particles, 
        hidden_dim = hidden_dim,
    )
    replan_agent_1.load(best_model_number_2, global_model_dir_3)

    replan_agent_2 = MADDPG(
        num_agents = n_particles, 
        state_dims = state_dim, 
        action_dims = action_dim, 
        min_action = min_action,
        max_action = max_action,         
        critic_input_dim = 9*n_particles, 
        hidden_dim = hidden_dim,
    )
    replan_agent_2.load(best_model_number_2, global_model_dir_4)

    delta_time_3 = 0.1 * math.sqrt(3)
    success_num = 100
    for n in range(10):  
        print(f'-----------------------The {n} th set of paths-----------------------')  

        original_paths, paths, failure1 = generate_global_paths(global_env, main_agent, n_particles, max_timesteps_1, levitator)            

        if failure1:
            print('-----------------------New planner failure!-----------------------')
            re_plan_paths, failure2 = generate_replan_paths(replan_env, replan_agent, n_particles, max_timesteps_2, original_paths)
            paths = re_plan_paths

            if failure2:
                re_plan_paths_1, failure3 = generate_replan_paths(replan_env, replan_agent_1, n_particles, max_timesteps_2, original_paths)
                paths = re_plan_paths_1

                if failure3:
                    re_plan_paths_2, failure4 = generate_replan_paths(replan_env, replan_agent_2, n_particles, max_timesteps_2, original_paths)
                    paths = re_plan_paths_2

                    if failure4:
                        print('Faliure!/n')
                        success_num -= 1
                        continue
                    else:
                        print('4th planner success!')
                else:
                    print('3rd planner success!')
            else:
                print('2nd planner success!')

        show(n_particles, np.transpose(original_paths, (1, 0, 2)))
        show(n_particles, np.transpose(paths, (1, 0, 2)))

    print(f'The success number: {success_num}')