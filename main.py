import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import argparse
from tqdm import tqdm
from atari_wrappers import wrap_deepmind, make_atari

total_rewards = []


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.pos = 0

    def insert(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.pos] = [state, action, reward, next_state, done]
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, done = zip(*batch)
        return observations, actions, rewards, next_observations, done

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(8, 8), stride=(4, 4))
        nn.init.trunc_normal_(self.conv1.weight, 1e-2)
        nn.init.constant_(self.conv1.bias, 1e-2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2))
        nn.init.trunc_normal_(self.conv2.weight, 1e-2)
        nn.init.constant_(self.conv2.bias, 1e-2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1))
        nn.init.trunc_normal_(self.conv3.weight, 1e-2)
        nn.init.constant_(self.conv3.bias, 1e-2)
        self.flat = nn.Flatten(-3)
        self.fc1 = nn.Linear(3136, 512)
        nn.init.trunc_normal_(self.fc1.weight, 1e-2)
        nn.init.constant_(self.fc1.bias, 1e-2)
        self.fc2 = nn.Linear(512, 4)
        nn.init.trunc_normal_(self.fc2.weight, 1e-2)
        nn.init.constant_(self.fc2.bias, 1e-2)

    def forward(self, states):
        x = F.relu(self.conv1(states))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        return q_values


class Agent():
    def __init__(self, env, start_epsilon=1.0, learning_rate=0.00025, GAMMA=0.99, batch_size=32, capacity=20000):
        self.env = env
        self.n_actions = 4
        self.count = 0

        self.epsilon = start_epsilon
        self.start_eps = start_epsilon
        self.end_eps = 0.1
        self.learning_rate = learning_rate
        self.gamma = GAMMA
        self.batch_size = batch_size
        self.capacity = capacity

        self.buffer = ReplayBuffer(self.capacity)
        self.evaluate_net = Net().cuda()
        self.target_net = Net().cuda()

        self.optimizer = torch.optim.RMSprop(
            self.evaluate_net.parameters(),
            lr=self.learning_rate,
            eps=1e-6,
            momentum=0,
            weight_decay=0.99,
        )

    def learn(self):
        if self.count % 5000 == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())

        observations, actions, rewards, next_observations, done = self.buffer.sample(self.batch_size)

        next_states = torch.tensor(numpy.array(next_observations), dtype=torch.float).cuda()
        next_states = torch.permute(next_states, (0, 3, 1, 2))
        re = torch.tensor(rewards, dtype=torch.float).view(self.batch_size, -1).cuda()
        acts = torch.tensor(actions, dtype=torch.long).view(self.batch_size, -1).cuda()

        not_done = [a[0] for a in enumerate(done) if a[1] == 0]
        not_done_states = torch.cat([next_states[a] for a in not_done]).view(-1, 4, 84, 84)

        evaluate = self.evaluate_net(next_states).gather(1, acts)
        target = torch.zeros(self.batch_size).cuda()
        target[not_done] = self.target_net(not_done_states).max(1)[0].detach()

        expect = self.gamma * target.unsqueeze(1) + re

        loss = nn.MSELoss()
        out = loss(evaluate, expect)

        self.optimizer.zero_grad()
        out.backward()
        torch.nn.utils.clip_grad_norm_(loss.parameters(), 1)
        self.optimizer.step()

    def choose_action(self, state):
        with torch.no_grad():
            rng = np.random.default_rng()
            r = rng.random()

            if r < self.epsilon:
                action = rng.integers(0, self.n_actions)
            else:
                state = torch.permute(torch.FloatTensor(numpy.array(state)).cuda(), (2, 0, 1))
                Q = self.evaluate_net.forward(state).squeeze(0).detach()
                action = int(torch.argmax(Q).cpu().numpy())

            if self.epsilon > self.end_eps and self.count > 50000:
                self.epsilon -= (self.start_eps - self.end_eps) / 1000000

        return action

    def check_max_Q(self):
        state = numpy.array(self.env.reset())
        state = torch.permute(torch.FloatTensor(state).cuda(), (2, 0, 1))

        return torch.max(self.evaluate_net(state.squeeze(0).detach())).item()


def train(env):
    agent = Agent(env)
    episode = 100000
    rewards = []
    for _ in tqdm(range(episode)):
        state = env.reset()
        count = 0
        while True:
            agent.count += 1
            # env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.buffer.insert(state, int(action), reward,
                                next_state, int(done))
            count += reward
            if len(agent.buffer) >= 32:
                if agent.count % 4 == 0:
                    agent.learn()
            if done:
                rewards.append(count)
                break
            state = next_state
    total_rewards.append(rewards)
    torch.save(agent.evaluate_net.state_dict(), "./Tables/DDQN.pt")


def test(env):
    rewards = []
    testing_agent = Agent(env)
    testing_agent.target_net.load_state_dict(torch.load("./Tables/DDQN.pt"))
    for _ in range(100):
        state = env.reset()
        count = 0
        while True:
            env.render()
            state = torch.permute(torch.FloatTensor(np.array(state)).cuda(), (2, 0, 1))
            Q = testing_agent.target_net.forward(state).squeeze(0).detach()
            action = int(torch.argmax(Q.cpu()).numpy())
            next_state, reward, done, _ = env.step(action)
            count += reward
            if done:
                rewards.append(count)
                break
            state = next_state
    print(f"reward: {np.mean(rewards)}")
    np.save("./Rewards/DDQN_test_rewards.npy", np.array(rewards))



def seed(seed=20):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    SEED = 2206070040

    env = wrap_deepmind(make_atari("BreakoutNoFrameskip-v4", 10000), frame_stack=True)
    seed(SEED)
    env.seed(SEED)
    env.action_space.seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    if not os.path.exists("./Tables"):
        os.mkdir("./Tables")

    if not os.path.exists("./Rewards"):
        os.mkdir("./Rewards")

    if args.train:
        for i in range(1):
            print(f"#{i + 1} training progress")
            train(env)
        np.save("./Rewards/DDQN_new_rewards.npy", np.array(total_rewards))

    test(env)

    env.close()
