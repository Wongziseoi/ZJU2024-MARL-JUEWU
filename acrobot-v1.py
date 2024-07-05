import gym
import numpy as np
import time

class Policy():
    def __init__(self, s_size=4, a_size=2):
        self.w = 1e-4*np.random.rand(s_size, a_size)  # weights for simple linear policy: state_space x action_space
        
    def forward(self, state):
        x = np.dot(state, self.w)
        return np.exp(x)/sum(np.exp(x))
    
    def act(self, state):
        probs = self.forward(state)
        action = np.argmax(probs)          
        return action

    def copy(self):
        new_policy = Policy()
        new_policy.w = self.w.copy()
        return new_policy

def run_episode(env, policy, render=False):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        # choose a random one in argmax scores
        if render:
            env.render()
        action = policy.act(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward

def hill_climbing(env, s_size, a_size, n_iterations=10000, noise_scale=0.3):
    best_reward = float('-inf')
    best_policy = Policy(s_size, a_size)

    for _ in range(n_iterations):
        # 提出优化方案
        new_policy = Policy(s_size, a_size)
        new_policy.w = best_policy.w + np.random.normal(0, noise_scale, (s_size, a_size))

        # 评估
        reward = np.mean([run_episode(env, new_policy) for _ in range(10)])

        if reward > best_reward:
            # （实际进行）优化
            best_policy = new_policy
            best_reward = reward

        print(f"Iteration {_}, best reward: {best_reward}")

        if best_reward >= -80:  # CartPole-v1的最大奖励是500
            break

    return best_policy, best_reward

env = gym.make('Acrobot-v1')

# 一共 3 个 actions，每个状态可以由 6 个特征表示
best_policy, best_reward = hill_climbing(env, s_size = 6, a_size = 3)

print(f"Best reward: {best_reward}")
print(f"Best weights: {best_policy.w}")

# 评估最终策略
final_rewards = [run_episode(env, best_policy) for _ in range(100)]
print(f"Average reward of final policy: {np.mean(final_rewards)}")

# 进行最终测试以及可视化
state=env.reset()
done = False
run_episode(env, best_policy, render=True)