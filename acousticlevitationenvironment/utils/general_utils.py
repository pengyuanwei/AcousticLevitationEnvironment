from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import gymnasium as gym
import copy
import argparse
import os
import shutil


# DDPG & MADDPG
class ReplayBufferDDPG:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)


# TD3
class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)    

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


# DQD3  
class ReplayBufferDQD3(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.action2 = np.zeros((max_size, action_dim))
        self.next_state2 = np.zeros((max_size, state_dim))
        self.reward2 = np.zeros((max_size, 1))
        self.not_done2 = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done, action2, next_state2, reward2, done2):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.action2[self.ptr] = action2
        self.next_state2[self.ptr] = next_state2
        self.reward2[self.ptr] = reward2
        self.not_done2[self.ptr] = 1. - done2

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)    

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.action2[ind]).to(self.device),
            torch.FloatTensor(self.next_state2[ind]).to(self.device),
            torch.FloatTensor(self.reward2[ind]).to(self.device),
            torch.FloatTensor(self.not_done2[ind]).to(self.device)
        )
    

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
            agent.save(i)
    return return_list


# DDPG
def train_off_policy_agent_DDPG(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(20):
        with tqdm(total=int(num_episodes/20), desc='Iter %d' % i) as pbar:
            for i_episode in range(int(num_episodes/20)):
                episode_return = 0
                state, info = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, truncated, info = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/20 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
            agent.save(i)
            evaluation(env, agent, i)
    return return_list


# TD3
def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(20):
        with tqdm(total=int(num_episodes/20), desc='Iter %d' % i) as pbar:
            for i_episode in range(int(num_episodes/20)):
                episode_return = 0
                state, info = env.reset()
                #print(state)
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, truncated, info = env.step(action)
                    replay_buffer.add(state, action, next_state, reward, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size > minimal_size:
                        agent.train(replay_buffer)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/20 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
            agent.save(i)
            #eval_agent(eval_env, agent)
    return return_list


# TD3 experiment
def train_off_policy_agent_experiment(env, agent, replay_buffer, minimal_size, total_timesteps, env_name, max_timesteps, i):
    num_evaluate = 10
    num_timesteps = 0
    return_list = []
    std_list = []
    num = 0
    while num_timesteps < total_timesteps:
        # Evaluate every 5000 time steps, each evaluation reports the average reward over 10 episodes with no exploration noise.
        # The results are reported over 10 random seeds of the Gym simulator and the network initialization.
        if num == 0 or num >= 5000:
            num = 0
            average_episode_return, return_std = eval_policy(agent, env_name, num_evaluate, max_timesteps)
            return_list.append(average_episode_return) 
            std_list.append(return_std)       
        
        state, info = env.reset()
        done = False
        while not done and num < 5000:
            if replay_buffer.size < minimal_size:
                action = env.action_space.sample()
            else:
                action = agent.take_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            replay_buffer.add(state, action, next_state, reward, done)
            state = next_state
            if replay_buffer.size > minimal_size:
                agent.train(replay_buffer)
            num_timesteps += 1
            if num_timesteps % (total_timesteps/100) == 0:
                percentage = num_timesteps/(total_timesteps/100)
                print("The No.", i, "th training has been finished:", percentage, "%.\n")
            if num_timesteps >= total_timesteps:
                break
            num += 1

    # print(len(return_list))
    agent.save_experiment(env_name, i)
    return return_list, std_list


# DQD3 experiment
def DQD3_train_off_policy_agent_experiment(env, agent, replay_buffer, minimal_size, total_timesteps, env_name, max_timesteps, i):
    num_evaluate = 10
    num_timesteps = 0
    return_list = []
    std_list = []
    while num_timesteps < total_timesteps:
        # Evaluate every 5000 time steps, each evaluation reports the average reward over 10 episodes with no exploration noise.
        # The results are reported over 10 random seeds of the Gym simulator and the network initialization.
        average_episode_return, return_std = eval_policy(agent, env_name, num_evaluate, max_timesteps)
        return_list.append(average_episode_return) 
        std_list.append(return_std)       
        
        num = 0
        state, info = env.reset()
        done = False
        while not done and num < 5000:
            if replay_buffer.size < minimal_size:
                action = env.action_space.sample()
                action2 = env.action_space.sample()
            else:
                action = agent.take_action(state, explore_mode=1)
                action2 = agent.take_action(state, explore_mode=2)
            env2 = copy.deepcopy(env)
            next_state, reward, done, truncated, info = env.step(action)
            next_state2, reward2, done2, truncated2, info2 = env2.step(action2)
            replay_buffer.add(state, action, next_state, reward, done, action2, next_state2, reward2, done2)
            state = next_state
            if replay_buffer.size > minimal_size:
                agent.train(replay_buffer)
            num_timesteps += 1
            if num_timesteps % (total_timesteps/100) == 0:
                percentage = num_timesteps/(total_timesteps/100)
                print("The No.", i, "th training has been finished:", percentage, "%.\n")
            if num_timesteps >= total_timesteps:
                break
            num += 1

    # print(len(return_list))
    agent.save_experiment(env_name, i)
    return return_list, std_list


# Independent TD3, MultiParticle_v5
def train_off_policy_agent_v1(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(20):
        with tqdm(total=int(num_episodes/20), desc='Iter %d' % i) as pbar:
            for i_episode in range(int(num_episodes/20)):
                episode_return = 0
                state, info = env.reset()
                #print(state)
                done = False
                while not done:
                    action0 = agent.take_action(state[0])
                    action1 = agent.take_action(state[1])
                    action2 = agent.take_action(state[2])
                    action3 = agent.take_action(state[3])

                    next_state, reward, done, truncated, info = env.step([action0, action1, action2, action3])
                    replay_buffer.add(state[0], action0, next_state[0], reward[0], done)
                    replay_buffer.add(state[1], action1, next_state[1], reward[1], done)
                    replay_buffer.add(state[2], action2, next_state[2], reward[2], done)
                    replay_buffer.add(state[3], action3, next_state[3], reward[3], done)

                    state = next_state
                    episode_return += sum(reward)
                    if replay_buffer.size > minimal_size:
                        agent.train(replay_buffer)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/20 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
            agent.save(i)
            #eval_agent(eval_env, agent)
    return return_list


# Independent TD3, MultiParticle_v1
def train_off_policy_agent_v3(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, save_dir):
    return_list = []
    for i in range(20):
        with tqdm(total=int(num_episodes/20), desc='Iter %d' % i) as pbar:
            for i_episode in range(int(num_episodes/20)):
                episode_return = 0
                state, info = env.reset()
                #print(state)
                done = False
                while not done:
                    action0 = agent.take_action(state[0])
                    action1 = agent.take_action(state[1])
                    action2 = agent.take_action(state[2])
                    action3 = agent.take_action(state[3])
                    action4 = agent.take_action(state[4])
                    action5 = agent.take_action(state[5])
                    action6 = agent.take_action(state[6])
                    action7 = agent.take_action(state[7])

                    next_state, reward, done, truncated, info = env.step([action0, action1, action2, action3, action4, action5, action6, action7])
                    replay_buffer.add(state[0], action0, next_state[0], reward[0], done)
                    replay_buffer.add(state[1], action1, next_state[1], reward[1], done)
                    replay_buffer.add(state[2], action2, next_state[2], reward[2], done)
                    replay_buffer.add(state[3], action3, next_state[3], reward[3], done)
                    replay_buffer.add(state[4], action4, next_state[4], reward[4], done)
                    replay_buffer.add(state[5], action5, next_state[5], reward[5], done)
                    replay_buffer.add(state[6], action6, next_state[6], reward[6], done)
                    replay_buffer.add(state[7], action7, next_state[7], reward[7], done)

                    state = next_state
                    episode_return += sum(reward)
                    if replay_buffer.size > minimal_size:
                        agent.train(replay_buffer, batch_size)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/20 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
            agent.save(i, save_dir)
            #eval_agent(eval_env, agent)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


# DDPG
def evaluation(env, agent, i):
    the_return = 0
    
    observation, info = env.reset(seed=42)
    for _ in range(1000):
        action = agent.take_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        
        the_return += reward
        
        #env.render()

        if terminated or truncated:
            observation, info = env.reset()
    
    if the_return >= agent.best_return:
        agent.best_return = the_return
        agent.best_model = i+1
        agent.save_best_model()


# TD3
# Runs policy/agent for X episodes and returns average reward
# Different seeds are used for the eval environment
def eval_policy(agent, env_name, eval_episodes=10, max_timesteps=1000):
    eval_env = gym.make(env_name)

    avg_reward = np.zeros([eval_episodes])
    for i in range(eval_episodes):
        random.seed(10*(i+1))
        np.random.seed(10*(i+1))
        torch.manual_seed(10*(i+1))

        num_timsteps = 0
        state, info = eval_env.reset()
        done = False
        while not done and num_timsteps < max_timesteps:
            action = agent.take_action(state, explore=False)
            next_state, reward, done, truncated, info = eval_env.step(action)
            state = next_state
            avg_reward[i] += reward
            num_timsteps += 1

    average_reward = np.mean(avg_reward)
    reward_std = np.std(avg_reward, ddof=1)

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {average_reward:.3f} +- {reward_std:.3f} ")
    print("---------------------------------------")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    return average_reward, reward_std


def check_dir(dir):
    # check save dir exists
    if os.path.isdir(dir):
        str_input = input("Save Directory already exists, would you like to continue (y,n)? ")
        if not str2bool(str_input):
            exit()
        else:
            # clear out existing files
            empty_dir(dir)


def empty_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")