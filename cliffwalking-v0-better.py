import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import gym
from tqdm import trange
import time


class Configs:
    def __init__(self, 
                 env, 
                 max_timestep=300, 
                 num_episode=10000,
                 plot_every=100,
                 alpha=0.4,
                 gamma=0.99,
                 epsilon_start=0.1,
                 epsilon_end=0.01,
                 decay_rate=0.99
                 ):
        self.obs_size = env.observation_space.n
        self.act_size = env.action_space.n

        self.max_timestep = max_timestep
        self.num_episode = num_episode
        self.plot_every = plot_every

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_rate = decay_rate


def update_q_value(q, q_next, reward, alpha, gamma):
    """
    TODO: 
    Please fill in the blank for variable 'td_target'
    according to the definition of the TD method
    """
    td_target = reward + gamma * q_next
    return q + alpha * (td_target - q)


def epsilon_greedy(q, timestep, epsilon_start=None, epsilon_end=None, decay_rate=None):
    epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-decay_rate * timestep)
    
    # 计算概率向量
    prob = np.ones_like(q) * epsilon / q.shape[0]
    prob[np.argmax(q)] = (1 - epsilon) + epsilon / q.shape[0]
    
    return prob


def sarsa(env, config):
    # initialize action-value function (empty dictionary of arrays)
    # q_values = np.zeros([config.obs_size, config.act_size], dtype=np.float32)
    q_values = np.random.rand(config.obs_size, config.act_size)

    # initialize performance monitor
    scores = deque(maxlen=config.plot_every)
    avg_scores = deque(maxlen=config.num_episode)

    # loop over episodes
    for episode in trange(1, config.num_episode + 1):
        reward_sum = 0
        state = env.reset()
        action_set = np.arange(config.act_size)

        prob = epsilon_greedy(q_values[state], episode, config.epsilon_start, config.epsilon_end, config.decay_rate)
        action = np.random.choice(action_set, p=prob)

        # 至多执行 max_timestep 步
        # 如果提前到达终点，也会提前结束
        for timestep in np.arange(config.max_timestep):
            env_feedbacks = env.step(action)
            next_state, reward, done = env_feedbacks[0], env_feedbacks[1], env_feedbacks[2]

            reward_sum += reward

            next_prob = epsilon_greedy(q_values[next_state], episode, config.epsilon_start, config.epsilon_end, config.decay_rate)
            next_action = np.random.choice(action_set, p=next_prob)
            next_q_value = q_values[next_state, next_action]
            q_values[state, action] = update_q_value(
                q_values[state, action], 
                0 if done else next_q_value, 
                reward, config.alpha, config.gamma)

            if done:break

            state, action = next_state, next_action

        scores.append(reward_sum)
                
        if episode % config.plot_every == 0:
            avg_scores.append(np.mean(scores))
    
    plt.plot(np.linspace(0, config.num_episode, len(avg_scores), endpoint=False), avg_scores, label="SARSA")
    plt.xlabel('Episode Number')
    plt.ylabel(f'Average Reward (Over Next {config.plot_every} Episodes)')

    print(f"SARSA Best Average Reward over {config.plot_every} Episodes: ", np.max(scores))
        
    return q_values


def q_learning(env, config):
    # initialize action-value function (empty dictionary of arrays)
    # q_values = np.zeros([config.obs_size, config.act_size], dtype=np.float32)
    q_values = np.random.rand(config.obs_size, config.act_size)

    # initialize performance monitor
    scores = deque(maxlen=config.plot_every)
    avg_scores = deque(maxlen=config.num_episode)

    # loop over episodes
    for episode in trange(1, config.num_episode + 1):
        reward_sum = 0
        state = env.reset()
        action_set = np.arange(config.act_size)

        prob = epsilon_greedy(q_values[state], episode, config.epsilon_start, config.epsilon_end, config.decay_rate)
        action = np.random.choice(action_set, p=prob)
        
        # 至多执行 max_timestep 步
        # 如果提前到达终点，也会提前结束
        for timestep in np.arange(config.max_timestep):      
            # 进行一步  
            env_feedbacks = env.step(action)
            next_state, reward, done = env_feedbacks[0], env_feedbacks[1], env_feedbacks[2]

            # 累加奖励
            reward_sum += reward

            # 计算出下一步的动作
            next_prob = epsilon_greedy(q_values[next_state], episode, config.epsilon_start, config.epsilon_end, config.decay_rate)
            next_action = np.random.choice(action_set, p=next_prob)

            # 更新 Q 值
            next_q_value = np.max(q_values[next_state])
            q_values[state, action] = update_q_value(
                    q_values[state, action], 
                    0 if done else next_q_value,
                    reward, config.alpha, config.gamma)

            if done:break
                
            state, action = next_state, next_action

        scores.append(reward_sum)
                        
        if episode % config.plot_every == 0:
            avg_scores.append(np.mean(scores))
    
    plt.plot(np.linspace(0, config.num_episode, len(avg_scores), endpoint=False), avg_scores, label="Q-learning")
    plt.xlabel('Episode Number')
    plt.ylabel(f'Average Reward (Over Next {config.plot_every} Episodes)')
    
    print(f"Q-Learning Best Average Reward over {config.plot_every} Episodes: ", np.max(scores))

    return q_values


if __name__ == "__main__":
    env = gym.make('CliffWalking-v0')
    print(env.action_space)
    print(env.observation_space)

    config = Configs(env, max_timestep=200, num_episode=10000, plot_every=100, epsilon_start=0.01, epsilon_end=0.001, decay_rate=0.99)
    
    # train the Q-learning and SARSA agent
    q_values_q_learning = q_learning(env, config)
    q_values_sarsa = sarsa(env, config)

    # print the optimal policy
    print("Q-Learning Optimal Policy:")
    print(np.argmax(q_values_q_learning, axis=1).reshape(4, 12))
    print("SARSA Optimal Policy:")
    print(np.argmax(q_values_sarsa, axis=1).reshape(4, 12))

    # plot the curves
    plt.legend()
    plt.show()