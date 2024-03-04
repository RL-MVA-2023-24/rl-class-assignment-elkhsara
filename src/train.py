from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import random
import os




env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.
# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dim = env.observation_space.shape[0]
nb_actions = env.action_space.n

config = {
    "nb_actions": nb_actions,
    "learning_rate": 0.001,
    "gamma": 0.95,
    "buffer_size": 1000000,
    "epsilon_min": 0.01,
    "epsilon_max": 1.0,
    "epsilon_decay_period": 10000,
    "epsilon_delay_decay": 400,
    "batch_size": 500,
    "gradient_steps": 2,
    "update_target_strategy": "ema",  # or 'replace'
    "update_target_freq": 600,
    "update_target_tau": 0.001,
    "criterion": torch.nn.SmoothL1Loss(),
    "monitoring_nb_trials": 50
}


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity)  # capacity of the buffer
        self.data = []
        self.index = 0  # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x: torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)


def greedy_action(network, state):
        device = "cuda" if next(network.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = network(torch.Tensor(state).unsqueeze(0).to(device))
            return torch.argmax(Q).item()


class DQN(torch.nn.Module):
    def __init__(self, state_dim =state_dim, nb_actions = nb_actions, num_layers=4, hidden_dim=512):
        super(DQN, self).__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(state_dim, hidden_dim)])
        self.layers.extend([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])
        self.output_layer = torch.nn.Linear(hidden_dim, nb_actions)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output_layer(x)



class ProjectAgent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DQN().to(self.device)
        self.nb_actions = config["nb_actions"]
        self.gamma = config["gamma"] if "gamma" in config.keys() else 0.95
        self.batch_size = config["batch_size"] if "batch_size" in config.keys() else 100
        buffer_size = config["buffer_size"] if "buffer_size" in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size, device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config["epsilon_min"] if "epsilon_min" in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        self.monitoring_nb_trials = config['monitoring_nb_trials'] if 'monitoring_nb_trials' in config.keys() else 0

    def act(self, state):
        return greedy_action(self.model, state)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        path = os.getcwd() + "/last_DQN.pt"
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.eval()

    def MC_eval(self, env, nb_trials):
        MC_total_reward = []
        MC_discounted_reward = []
        for _ in range(nb_trials):
            x, _ = env.reset()
            done = False
            trunc = False
            total_reward = 0
            discounted_reward = 0
            step = 0
            while not (done or trunc):
                a = greedy_action(self.model, x)
                y, r, done, trunc, _ = env.step(a)
                x = y
                total_reward += r
                discounted_reward += self.gamma**step * r
                step += 1
            MC_total_reward.append(total_reward)
            MC_discounted_reward.append(discounted_reward)
        return np.mean(MC_discounted_reward), np.mean(MC_total_reward)

    def V_initial_state(self, env, nb_trials):
        with torch.no_grad():
            for _ in range(nb_trials):
                val = []
                x, _ = env.reset()
                val.append(
                    self.model(torch.Tensor(x).unsqueeze(0).to(device)).max().item()
                )
        return np.mean(val)

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1 - D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, env, max_episode):
        episode_return = []
        MC_avg_total_reward = []
        MC_avg_discounted_reward = []
        V_init_state = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        best = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps):
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == "replace":
                if step % self.update_target_freq == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == "ema":
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1 - tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                # Monitoring
                if self.monitoring_nb_trials > 0 and episode % 200 == 0 :
                    MC_dr, MC_tr = self.MC_eval(env, self.monitoring_nb_trials)
                    V0 = self.V_initial_state(env, self.monitoring_nb_trials)
                    MC_avg_total_reward.append(MC_tr)
                    MC_avg_discounted_reward.append(MC_dr)
                    V_init_state.append(V0)
                    episode_return.append(episode_cum_reward)
                    print("Episode ", '{:2d}'.format(episode),
                          ", epsilon ", '{:6.2f}'.format(epsilon),
                          ", batch size ", '{:4d}'.format(len(self.memory)),
                          ", ep return ", '{:4.1f}'.format(episode_cum_reward),
                          ", MC tot ", '{:6.2f}'.format(MC_tr),
                          ", MC disc ", '{:6.2f}'.format(MC_dr),
                          ", V0 ", '{:6.2f}'.format(V0),
                          sep='')

                    if MC_tr > best:
                        best = MC_tr
                        print("Best score is", best)
                        torch.save(self.model.state_dict(),f"DQN")
                        print("Best Model saved")
                else:
                    episode_return.append(episode_cum_reward)
                    print("Episode ", '{:2d}'.format(episode),
                          ", epsilon ", '{:6.2f}'.format(epsilon),
                          ", batch size ", '{:4d}'.format(len(self.memory)),
                          ", ep return ", '{:4.1f}'.format(episode_cum_reward),
                          sep='')

                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return, MC_avg_discounted_reward, MC_avg_total_reward, V_init_state


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)



#seed_everything(seed=42)
#agent = ProjectAgent()
#print("Start training")
#ep_length, disc_rewards, tot_rewards, V0 = agent.train(env, 200)
#agent.save("./last_DQN.pt")
#print("Training done")