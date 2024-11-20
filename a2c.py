import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Create the Flappy Bird environment
env = gym.make("FlappyBird-v0")  # Replace with your environment name

# Wrap the environment in a DummyVecEnv for compatibility with Stable-Baselines3
env = DummyVecEnv([lambda: env])

# Initialize the A2C model
model = A2C("MlpPolicy", env, verbose=1, learning_rate=1e-3, gamma=0.99)

# Train the model
model.learn(total_timesteps=100000)

# Save the trained model
model.save("a2c_flappy_bird")

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

# Load the model for future use
model = A2C.load("a2c_flappy_bird")











# import torch
# from torch import nn
# import torch.optim as optim
# from torch.distributions import Categorical
# from torch.utils.tensorboard import SummaryWriter
# import numpy as np
# import os


# class A2C(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=256):
#         super(A2C, self).__init__()
#         self.shared = nn.Sequential(
#             nn.Linear(state_dim, hidden_dim),
#             nn.ReLU()
#         )
#         self.actor = nn.Linear(hidden_dim, action_dim)
#         self.critic = nn.Linear(hidden_dim, 1)
        
#     def forward(self, x):
#         x = self.shared(x)
#         policy = torch.softmax(self.actor(x), dim=-1)
#         value = self.critic(x)
#         return policy, value
    
# class A2CAgent:
#     def __init__(self, env, learning_rate=1e-3, gamma=0.99, log_dir='runs/a2c_logs'):
#         self.env = env
#         self.gamma = gamma
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.state_dim = env.observation_space.shape[0]
#         self.action_dim = env.action_space.n
        
#         self.model = A2C(self.state_dim, self.action_dim).to(self.device)
#         self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
#         self.writer = SummaryWriter(log_dir=log_dir)
        
#     def select_action(self, state):
#         if not isinstance(state, torch.Tensor):  # Ensure input is a tensor
#             state = torch.tensor(state, dtype=torch.float32, device=self.device)
#         policy, _ = self.model(state)
#         action_dist = Categorical(policy)
#         action = action_dist.sample()
#         return action.item(), action_dist.log_prob(action)
    
#     def compute_returns(self, rewards):
#         returns = []
#         g = 0
#         for reward in reversed(rewards):
#             g = reward + self.gamma * g
#             returns.insert(0, g)
#         return torch.tensor(returns, dtype=torch.float32, device=self.device)

#     def update_policy(self, log_probs, values, returns):
#         values = torch.cat(values)
#         log_probs = torch.cat(log_probs)
#         returns = (returns - returns.mean()) / (returns.std() + 1e-8)

#         advantages = returns - values.detach()
#         policy_loss = -(log_probs * advantages).mean()
#         value_loss = (returns - values).pow(2).mean()

#         loss = policy_loss + value_loss
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         return loss.item()
    
#     def train(self, episodes=500_000, render=False):
#         for episode in range(episodes):
#             state, _ = self.env.reset()
#             print(f"Initial observation: {state}, Expected shape: {self.env.observation_space.shape}")
#             state = torch.tensor(state, dtype=torch.float32, device=self.device)
#             log_probs, values, rewards = [], [], []
#             episode_reward = 0

#             done = False
#             while not done:
#                 if render:
#                     self.env.render()

#                 action, log_prob = self.select_action(state)
#                 log_probs.append(log_prob.unsqueeze(0))
#                 next_state, reward, done, _, _ = self.env.step(action)
#                 print(f"Next observation: {next_state}, Reward: {reward}, Done: {done}, Expected shape: {self.env.observation_space.shape}")
                
#                 if isinstance(next_state, np.ndarray):
#                     next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)


#                 _, value = self.model(torch.tensor(state, dtype=torch.float32, device=self.device))
#                 log_probs.append(log_prob)
#                 values.append(value)
#                 rewards.append(reward)
#                 episode_reward += reward
#                 state = next_state

#             returns = self.compute_returns(rewards)
#             loss = self.update_policy(log_probs, values, returns)

#             # Log metrics
#             self.writer.add_scalar("Reward", episode_reward, episode)
#             self.writer.add_scalar("Loss", loss, episode)
            
#             if (episode + 1) % 10_000 == 0:
#                 self.save_model(f"runs/a2c_model_episode_{episode + 1}.pt")

#         # Final save and close TensorBoard writer
#         self.save_model("runs/a2c_model_final.pt")
#         self.writer.close()

#     def save_model(self, path="runs/a2c_model.pt"):
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         torch.save(self.model.state_dict(), path)
#         print(f"Model saved to {path}")

#     def load_model(self, path="runs/a2c_model.pt"):
#         if os.path.exists(path):
#             self.model.load_state_dict(torch.load(path))
#             self.model.eval()
#             print(f"Model loaded from {path}")
#         else:
#             print(f"No model found at {path}")

#     def test(self):
#         state, _ = self.env.reset()
#         done = False
#         while not done:
#             self.env.render()
#             action, _ = self.select_action(state)
#             state, _, done, _, _ = self.env.step(action)
        