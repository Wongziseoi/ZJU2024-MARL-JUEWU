import gym
import numpy as np
from collections import deque

import time


env = gym.make('CartPole-v1')
print('observation space:', env.observation_space)
print('action space:', env.action_space)
threshold = env.spec.reward_threshold
print('threshold: ', threshold)

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


env.seed(0)
np.random.seed(0)

policy = Policy()

def hill_climbing(n_episodes=10000, gamma=0.99, noise_scale=1e-2):

    scores_deque = deque(maxlen=100)
    scores = []
    arr_noise = []
    best_Gt = -np.Inf
    best_w = policy.w
    for i_episode in range(1, n_episodes+1):
        rewards = []
        state = env.reset()
        timesteps = 0 ## is the same as len(rewards)
        
        while True:
            if i_episode % 100 == 0:
                time.sleep(0.01)
                env.render()
                pass

            action = policy.act(state)
            state, reward, done, _ = env.step(action)            
            rewards.append(reward)
            timesteps += 1  
            if done:
                break 
                
        total_reward = sum(rewards)        
        scores_deque.append(total_reward)
        scores.append(total_reward)

        discounts = [gamma**i for i in range(len(rewards)+1)]
        ## This is the 'cumulative discounted reward' or TD-target G_t
        Gt = sum([a*b for a,b in zip(discounts, rewards)])
        
        if Gt >= best_Gt: # found better weights
            ## if Gt > best_R ==> decrease the noise: noise = noise/2  (till 0.001)
            best_Gt = Gt
            best_w = policy.w
            noise_scale = max(1e-3, noise_scale / 2)
            print('Ep.: {:3d} , timesteps: {:3d}, noise_scale (Gt >= best(Gt)): {:.4f}, Gt {:.4f}, \tAvg.Score:  {:.3f}'\
                  .format(i_episode, timesteps, noise_scale, Gt, np.mean(scores_deque)))
            policy.w += noise_scale * np.random.rand(*policy.w.shape) 
        else: # did not find better weights
            ## if Gt < best_R ==> increase the noise: noise = 2*noise (till 2)
            noise_scale = min(2, noise_scale * 2)
            print('Ep.: {:3d} , timesteps: {:3d}, noise_scale (Gt < best(Gt)): {:.4f}, Gt {:.4f}, \tAvg.Score:  {:.3f}'\
                  .format(i_episode, timesteps, noise_scale, Gt, np.mean(scores_deque)))
            policy.w = best_w + noise_scale * np.random.rand(*policy.w.shape)
            
        arr_noise.append(noise_scale)   

        if np.mean(scores_deque)>=threshold:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            policy.w = best_w
            break
    
    return scores, arr_noise
            
scores, arr_noise = hill_climbing()


state=env.reset()
done = False
while not done:
    time.sleep(0.01)
    env.render()
    action = policy.act(state)
    state, reward, done, _ = env.step(action)            
    