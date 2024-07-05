
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        '''
        TODO:
        conv1: 输入维度，输出维度:32，kernal_size:3, stride:2, padding:1
        conv2:输入维度：32，输出维度:32，kernal_size:3, stride:2, padding:1
        conv3:输入维度：32，输出维度:32，kernal_size:3, stride:2, padding:1
        conv4:输入维度：32，输出维度:32，kernal_size:3, stride:2, padding:1
        lstm:(LSTMCell)，输入维度：32*6*6，输出维度512
        critic：输入512，输出1
        actor：输入512，输出action_space
        '''

        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(32 * 6 * 6, 512)
        self.critic = nn.Linear(512, 1)
        self.actor = nn.Linear(512, num_actions)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, hx, cx):
        '''
        Input tensor shape: x.shape = (batch_size, num_inputs, height, width)
        Output tensor shapes:
        - Actor value: actor(x).shape = (batch_size, num_actions)
        - Critic value: critic(x).shape = (batch_size, 1)
        - Hidden state: hx.shape = (batch_size, 512)
        - Cell state: cx.shape = (batch_size, 512)
        '''
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        N = x.shape[0]
        x = x.view(N, -1) # flatten
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        return self.actor(x), self.critic(x), hx, cx



