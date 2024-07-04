import numpy as np
import torch


class ReplayBuffer:
	def __init__(self, state_dim, action_dim, max_size=int(1e6), device="cpu"):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, 1))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = device

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
			torch.tensor(self.state[ind], device=self.device, dtype=torch.float32),
			torch.tensor(self.action[ind], device=self.device, dtype=torch.int64),
			torch.tensor(self.next_state[ind], device=self.device, dtype=torch.float32),
			torch.tensor(self.reward[ind], device=self.device, dtype=torch.float32),
			torch.tensor(self.not_done[ind], device=self.device, dtype=torch.float32)
		)