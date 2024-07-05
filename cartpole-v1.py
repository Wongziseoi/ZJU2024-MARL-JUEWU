import gym
import numpy as np
import time

def linear_policy(weights, observation):
    return np.dot(weights, observation)

def run_episode(env, weights_left, weights_right):
    observation = env.reset()
    total_reward = 0
    done = False
    while not done:
        score_left = linear_policy(weights_left, observation)
        score_right = linear_policy(weights_right, observation)
        observation, reward, done, _ = env.step(0 if score_left > score_right else 1)
        total_reward += reward
    return total_reward

def hill_climbing(env, n_iterations=10000, noise_scale=0.01):
    weights_left = np.random.normal(0, noise_scale, size=(4))  # 初始随机权重
    weights_right = np.random.normal(0, noise_scale, size=(4))  # 初始随机权重
    best_reward = float('-inf')
    best_weights_left = weights_left.copy()
    best_weights_right = weights_right.copy()

    for _ in range(n_iterations):
        new_weights_left = weights_left + np.random.normal(0, noise_scale, 4)  # 添加噪声
        new_weights_right = weights_right + np.random.normal(0, noise_scale, 4)
        reward = np.mean([run_episode(env, new_weights_left, new_weights_right) for _ in range(10)])  # 评估新权重

        if reward > best_reward:
            best_reward = reward
            best_weights_left = new_weights_left.copy()
            best_weights_right = new_weights_right.copy()
            weights_left = best_weights_left.copy()  # 接受新权重
            weights_right = best_weights_right.copy()
            
        print(f"Iteration {_}, best reward: {best_reward}")

        if best_reward >= 500:  # CartPole-v1的最大奖励是500
            break

    return best_weights_left, best_weights_right, best_reward

env = gym.make('CartPole-v1')
best_weights_left, best_weights_right, best_reward = hill_climbing(env)

print(f"Best reward: {best_reward}")
print(f"Best weights left: {best_weights_left}")
print(f"Best weights right: {best_weights_right}")

# 评估最终策略
final_rewards = [run_episode(env, best_weights_left, best_weights_right) for _ in range(100)]
print(f"Average reward of final policy: {np.mean(final_rewards)}")

# 进行最终测试以及可视化
observation=env.reset()
done = False
while not done:
    time.sleep(0.001)
    env.render()
    score_left = linear_policy(best_weights_left, observation)
    score_right = linear_policy(best_weights_right, observation)
    observation, reward, done, _ = env.step(0 if score_left > score_right else 1)
