import os
import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import acousticlevitationgym

from acousticlevitationenvironment.utils import general_utils
from acoustorl import MADDPG
from acoustorl.common.general_utils import ReplayBuffer_MADDPG


def train_off_policy_agent_experiment(env, agent, replay_buffer, batch_size, minimal_size, total_timesteps, eval_env, max_timesteps, save_dir, n_particles, seed):
    num_evaluate = 50
    num_timesteps = 0
    best_episode_return = 0
    return_list = []
    best_success_rate = 0
    success_number_list = []
    eval_index = 0
    while num_timesteps < total_timesteps:
        
        state, info = env.reset(seed=seed)
        terminated, truncated = False, False

        while (not terminated) and (not truncated):
            if replay_buffer.memory_num < minimal_size:
                action = env.action_space.sample()
            else:
                action = agent.take_action(state)  
                
            next_state, reward, terminated, truncated, info = env.step(action)

            expert_action = env.expert_actions()

            replay_buffer.store(state, expert_action, reward, next_state, terminated)

            state = next_state

            if replay_buffer.memory_num > minimal_size:
                for i in range(n_particles):
                    agent.train(replay_buffer, batch_size, i)

            # Evaluate every 5000 time steps, each evaluation reports the average reward over 10 episodes with no exploration noise.
            # The results are reported over 10 random seeds of the Gym simulator and the network initialization.
            if num_timesteps % (total_timesteps/100) == 0:
                average_episode_return, success_rate = eval_policy(agent, eval_env, n_particles, num_evaluate, max_timesteps, seed)
                return_list.append(average_episode_return) 
                success_number_list.append(success_rate)

                if average_episode_return > best_episode_return:
                    best_episode_return = average_episode_return
                    agent.save(eval_index, save_dir)
                    agent.save(999, save_dir)
                    print("===============================================================================The new highest average reward!")
                if success_rate >= best_success_rate and success_rate != 0:
                    best_success_rate = success_rate
                    agent.save(eval_index, save_dir)
                    agent.save(1000, save_dir)
                    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++The new best success number!")

                percentage = num_timesteps/(total_timesteps/100)
                print("---------------------------------------")
                print("The", save_dir, "th training has been finished:", percentage, "%.")
                print("---------------------------------------")

                eval_index += 1

            num_timesteps += 1

    return return_list, success_number_list


def eval_policy(agent, eval_env, n_particles, eval_episodes, max_timesteps, seed=None):

    success_num = 0.0
    episode_reward = np.zeros([eval_episodes])
    for i in range(eval_episodes):

        state, info = eval_env.reset(seed=i)
        terminated, truncated = False, False

        while (not terminated) and (not truncated):
            action = agent.take_action(state, explore=False)

            next_state, reward, terminated, truncated, _ = eval_env.step(action)

            state = next_state

            episode_reward[i] += np.mean(reward)

            if terminated == True:
                success_num += 1

    average_reward = np.mean(episode_reward)
    #reward_std = np.std(episode_reward, ddof=1)

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} evaluated episodes: {average_reward:.3f}")
    print(f"The success number: {success_num} over {eval_episodes} evaluations")
    print("---------------------------------------")

    return average_reward, success_num


if __name__ == "__main__":

    save_dir = './experiments/experiment_205'
    n_particles = 6
    
    general_utils.check_dir(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    total_timesteps = 1000000
    delta_time = 1.0/10
    max_timesteps = 20
    buffer_size = 500000
    minimal_size = 25000
    batch_size = 512
    seed = None

    hidden_dim = 64
    exploration_noise = 0.3
    discount = 0.95
    tau = 0.005
    actor_lr = 2e-4
    critic_lr = 4e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("The device is:", device)

    env_name = "acoustic_levitation_environment_v2/GlobalTrain-v0"
    eval_env_name = "acoustic_levitation_environment_v2/GlobalPlanner-v0"

    env = gym.make(env_name, n_particles=n_particles, delta_time=delta_time, max_timesteps=max_timesteps)
    eval_env = gym.make(eval_env_name, n_particles=n_particles, delta_time=delta_time, max_timesteps=max_timesteps)

    state_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].shape[0]
    min_action = env.action_space[0].low[0]
    max_action = env.action_space[0].high[0]  # 动作最大值

    agent = MADDPG(
        num_agents = n_particles, 
        state_dims = state_dim, 
        action_dims = action_dim, 
        min_action = min_action,
		max_action = max_action,         
        critic_input_dim = 9*n_particles, 
        hidden_dim = hidden_dim,
        exploration_noise = exploration_noise,
        gamma = discount, 
        tau = tau,
        actor_lr = actor_lr, 
        critic_lr = critic_lr, 
        device = device
    )
    
    replay_buffer = ReplayBuffer_MADDPG(
        num_agents = n_particles,
        state_dim = state_dim,
        action_dim = action_dim,
        max_size = buffer_size,
        device = device
    )

    return_list, success_number_list = train_off_policy_agent_experiment(
        env, 
        agent, 
        replay_buffer,
        batch_size,
        minimal_size, 
        total_timesteps, 
        eval_env, 
        max_timesteps, 
        save_dir,
        n_particles,
        seed
    )


    iteration_list = list(range(len(return_list)))

    plt.plot(iteration_list, success_number_list, color=(0.0, 0.0, 1.0, 0.8), label=save_dir)
    plt.xlabel('Timesteps (1e4)')
    plt.ylabel('Success Number (per 20 evaluations)')
    plt.title('Learning curve of {}'.format(save_dir))
    plt.legend()
    plt.show()

    plt.plot(iteration_list, return_list, color=(0.0, 0.0, 1.0, 0.8), label=save_dir)
    plt.xlabel('Timesteps (1e4)')
    plt.ylabel('Average Reward (per 20 evaluations)')
    plt.title('Learning curve of {}'.format(save_dir))
    plt.legend()
    plt.show()

    np.save(save_dir + '/return_list.npy', return_list)   # 保存为.npy格式
    print("The length of return list is", len(return_list))
    np.save(save_dir + '/success_number_list.npy', success_number_list)   # 保存为.npy格式
    print("The length of success_number_list is", len(success_number_list))