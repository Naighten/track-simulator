import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import yaml


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class RLAgent:
    def __init__(self, env):
        with open("rl/config.yaml") as f:
            self.cfg = yaml.safe_load(f)

        self.env = env
        self.input_size = env.observation_space.shape[0]
        self.output_size = env.action_space.shape[0]

        self.model = NeuralNetwork(self.input_size, self.output_size)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.cfg["training"]["learning_rate"]
        )

    def load_neat_checkpoint(self, path):
        """Convert NEAT genome to neural network weights"""
        with open(path, "rb") as f:
            genome = pickle.load(f)

        # Простейшая конвертация - первые 10 соединений
        with torch.no_grad():
            for i, conn in enumerate(genome.connections.values()):
                if i >= 10: break
                self.model.net[0].weight.data[i] = conn.weight * 0.1

    def train(self):
        for episode in range(self.cfg["training"]["episodes"]):
            obs, _ = self.env.reset()
            total_reward = 0

            while True:
                obs_tensor = torch.FloatTensor(obs)
                action = self.model(obs_tensor).detach().numpy()

                next_obs, reward, done, _, _ = self.env.step(action)
                total_reward += reward

                # Пример обновления (дописать под конкретный алгоритм)
                loss = torch.mean(torch.square(self.model(obs_tensor)))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                obs = next_obs
                if done:
                    break

            print(f"Episode {episode} | Reward: {total_reward:.1f}")
