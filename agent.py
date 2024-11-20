import gymnasium as gym
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn
import yaml

from experience_replay import ReplayMemory
from dqn import DQN

from datetime import datetime, timedelta
import argparse
import itertools

import flappy_bird_gymnasium
import os

import argparse
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv


# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu' # force cpu, sometimes GPU not always faster than CPU due to overhead of moving data to GPU

def train_dqn():
    print("Training with DQN...")
    agent = DQN()
    agent.run(is_training=True)

def test_dqn(env_name="FlappyBird-v0"):
    print("Testing with DQN...")

    # Create the environment
    import gymnasium as gym
    env = gym.make(env_name, render_mode="human")

    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize the DQN agent
    agent = DQN(state_dim=state_dim, action_dim=action_dim)

    # Test the agent
    obs, _ = env.reset()
    done = False
    while not done:
        action = agent.select_action(obs)  # Use the trained model to select actions
        obs, reward, done, _, _ = env.step(action)
        env.render()

    env.close()


# def train_a2c(env_name="FlappyBird-v0", timesteps=10_000_000):
#     print("Training with A2C...")
#     env = DummyVecEnv([lambda: gym.make(env_name)])
#     model = A2C("MlpPolicy", env, verbose=1, learning_rate=1e-3, gamma=0.99, tensorboard_log="./a2c_logs/")
#     model.learn(total_timesteps=timesteps)
#     model.save("runs/a2c_flappy_bird.zip")
#     print("A2C model saved to runs/a2c_flappy_bird.zip")

def train_a2c(env_name="FlappyBird-v0", timesteps=10_000_000):
    print("Training with A2C...")

    # Create environment
    env = DummyVecEnv([lambda: gym.make(env_name)])

    # Initialize model
    model = A2C("MlpPolicy", env, verbose=1, learning_rate=1e-3, gamma=0.99)

    # Initialize metrics storage
    episode_rewards = []
    losses = []
    episodes = []

    # Initialize Matplotlib for live plotting
    plt.ion()
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].set_title("Episode Rewards")
    ax[0].set_xlabel("Episode")
    ax[0].set_ylabel("Reward")
    ax[1].set_title("Policy Loss")
    ax[1].set_xlabel("Episode")
    ax[1].set_ylabel("Loss")

    # Callback function for logging and plotting
    def custom_callback(_locals, _globals):
        nonlocal episodes, episode_rewards, losses

        if "infos" in _locals:
            infos = _locals["infos"]
            for info in infos:
                if "episode" in info.keys():
                    # Log rewards
                    episode_rewards.append(info["episode"]["r"])
                    episodes.append(len(episodes) + 1)

                    # Update reward plot
                    ax[0].plot(episodes, episode_rewards, color="blue")
                    ax[0].relim()
                    ax[0].autoscale_view()

        # Log loss (can be accessed from locals)
        if "policy_loss" in _locals:
            policy_loss = _locals["policy_loss"]
            losses.append(policy_loss)

            # Update loss plot
            ax[1].plot(episodes, losses, color="red")
            ax[1].relim()
            ax[1].autoscale_view()

        plt.draw()
        plt.pause(0.01)

        return True

    # Train model with custom callback for logging
    model.learn(total_timesteps=timesteps, callback=custom_callback)

    # Save model
    model.save("runs/a2c_flappy_bird")
    print("A2C model saved to runs/a2c_flappy_bird.zip")

    # Finalize plots
    plt.ioff()
    plt.show()

def test_a2c(env_name="FlappyBird-v0"):
    print("Testing with A2C...")
    # Add render_mode="human" for rendering
    env = DummyVecEnv([lambda: gym.make(env_name, render_mode="human")])
    model = A2C.load("runs/a2c_flappy_bird.zip")
    
    obs = env.reset()
    while True:
        # Predict action
        action, _ = model.predict(obs)
        # Step the environment
        obs, _, done, _ = env.step(action)
        if done:
            break


# Deep Q-Learning Agent
class Agent():

    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            # print(hyperparameters)

        self.hyperparameter_set = hyperparameter_set

        # Hyperparameters (adjustable)
        self.env_id             = hyperparameters['env_id']
        self.learning_rate_a    = hyperparameters['learning_rate_a']        # learning rate (alpha)
        self.discount_factor_g  = hyperparameters['discount_factor_g']      # discount rate (gamma)
        self.network_sync_rate  = hyperparameters['network_sync_rate']      # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
        self.stop_on_reward     = hyperparameters['stop_on_reward']         # stop training after reaching this number of rewards
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.env_make_params    = hyperparameters.get('env_make_params',{}) # Get optional environment-specific parameters, default to empty dict
        self.enable_double_dqn  = hyperparameters['enable_double_dqn']      # double dqn on/off flag

        # Neural Network
        self.loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
        self.optimizer = None                # NN Optimizer. Initialize later.

        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def run(self, is_training=True, render=False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        # Create instance of the environment.
        # Use "**self.env_make_params" to pass in environment-specific parameters from hyperparameters.yml.
        env = gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)

        # Number of possible actions
        num_actions = env.action_space.n

        # Get observation space size
        num_states = env.observation_space.shape[0] # Expecting type: Box(low, high, (shape0,), float64)

        # List to keep track of rewards collected per episode.
        rewards_per_episode = []

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_double_dqn).to(device)

        if is_training:
            # Initialize epsilon
            epsilon = self.epsilon_init

            # Initialize replay memory
            memory = ReplayMemory(self.replay_memory_size)

            # Create the target network and make it identical to the policy network
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_double_dqn).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # Policy network optimizer. "Adam" optimizer can be swapped to something else.
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            # List to keep track of epsilon decay
            epsilon_history = []

            # Track number of steps taken. Used for syncing policy => target network.
            step_count=0

            # Track best reward
            best_reward = -9999999
        else:
            # Load learned policy
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

            # switch model to evaluation mode
            policy_dqn.eval()

        # Train INDEFINITELY, manually stop the run when you are satisfied (or unsatisfied) with the results
        for episode in itertools.count():

            state, _ = env.reset()  # Initialize environment. Reset returns (state,info).
            state = torch.tensor(state, dtype=torch.float, device=device) # Convert state to tensor directly on device

            terminated = False      # True when agent reaches goal or fails
            episode_reward = 0.0    # Used to accumulate rewards per episode

            # Perform actions until episode terminates or reaches max rewards
            # (on some envs, it is possible for the agent to train to a point where it NEVER terminates, so stop on reward is necessary)
            while(not terminated and episode_reward < self.stop_on_reward):

                # Select action based on epsilon-greedy
                if is_training and random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    # select best action
                    with torch.no_grad():
                        # state.unsqueeze(dim=0): Pytorch expects a batch layer, so add batch dimension i.e. tensor([1, 2, 3]) unsqueezes to tensor([[1, 2, 3]])
                        # policy_dqn returns tensor([[1], [2], [3]]), so squeeze it to tensor([1, 2, 3]).
                        # argmax finds the index of the largest element.
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Execute action. Truncated and info is not used.
                new_state,reward,terminated,truncated,info = env.step(action.item())

                # Accumulate rewards
                episode_reward += reward

                # Convert new state and reward to tensors on device
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    # Save experience into memory
                    memory.append((state, action, new_state, reward, terminated))

                    # Increment step counter
                    step_count+=1

                # Move to the next state
                state = new_state

            # Keep track of the rewards collected per episode.
            rewards_per_episode.append(episode_reward)

            # Save model when new best reward is obtained.
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward


                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                # If enough experience has been collected
                if len(memory)>self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Decay epsilon
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    # Copy policy network to target network after a certain number of steps
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count=0


    # def save_graph(self, rewards_per_episode, epsilon_history):
    #     # Save plots
    #     fig = plt.figure(1)
    #
    #     # Plot average rewards (Y-axis) vs episodes (X-axis)
    #     mean_rewards = np.zeros(len(rewards_per_episode))
    #     for x in range(len(mean_rewards)):
    #         mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
    #     plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
    #     # plt.xlabel('Episodes')
    #     plt.ylabel('Mean Rewards')
    #     plt.plot(mean_rewards)
    #
    #     # Plot epsilon decay (Y-axis) vs episodes (X-axis)
    #     plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
    #     # plt.xlabel('Time Steps')
    #     plt.ylabel('Epsilon Decay')
    #     plt.plot(epsilon_history)
    #
    #     plt.subplots_adjust(wspace=1.0, hspace=1.0)
    #
    #     # Save plots
    #     fig.savefig(self.GRAPH_FILE)
    #     plt.close(fig)

    def save_graph(self, rewards_per_episode, epsilon_history):
        """
        Saves two separate plots for mean rewards and epsilon decay.
        Args:
            rewards_per_episode (list): List of rewards per episode.
            epsilon_history (list): List of epsilon values during training.
        """
        # Calculate mean rewards over a sliding window of 100 episodes
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - 99):(x + 1)])

        # Plot and save Mean Rewards graph
        plt.figure(figsize=(12, 6))  # Larger figure size for clarity
        plt.plot(mean_rewards, label="Mean Rewards")
        plt.xlabel("Episodes")
        plt.ylabel("Mean Rewards")
        plt.title("Mean Rewards per Episode")
        plt.legend()
        plt.grid(True)
        mean_rewards_path = "mean_rewards.png"
        plt.savefig(mean_rewards_path)
        plt.close()
        print(f"Mean rewards plot saved to {mean_rewards_path}")

        # Plot and save Epsilon Decay graph
        plt.figure(figsize=(12, 6))  # Larger figure size for clarity
        plt.plot(epsilon_history, label="Epsilon Decay", color='orange')
        plt.xlabel("Time Steps")
        plt.ylabel("Epsilon")
        plt.title("Epsilon Decay over Time")
        plt.legend()
        plt.grid(True)
        epsilon_decay_path = "epsilon_decay.png"
        plt.savefig(epsilon_decay_path)
        plt.close()
        print(f"Epsilon decay plot saved to {epsilon_decay_path}")

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Transpose the list of experiences and separate each element
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors
        # tensor([[1,2,3]])
        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)

        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            if self.enable_double_dqn:
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)

                target_q = rewards + (1-terminations) * self.discount_factor_g * \
                                target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                # Calculate target Q values (expected returns)
                target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
                '''
                    target_dqn(new_states)  ==> tensor([[1,2,3],[4,5,6]])
                        .max(dim=1)         ==> torch.return_types.max(values=tensor([3,6]), indices=tensor([3, 0, 0, 1]))
                            [0]             ==> tensor([3,6])
                '''

        # Calcuate Q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        '''
            policy_dqn(states)  ==> tensor([[1,2,3],[4,5,6]])
                actions.unsqueeze(dim=1)
                .gather(1, actions.unsqueeze(dim=1))  ==>
                    .squeeze()                    ==>
        '''

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model (backpropagation)
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients
        self.optimizer.step()       # Update network parameters i.e. weights and biases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test RL models for Flappy Bird.")
    parser.add_argument("--train", help="Training mode", action="store_true")
    parser.add_argument("--algorithm", choices=["dqn", "a2c"], required=True, help="Choose the RL algorithm")
    parser.add_argument("--hyperparameters", help="The hyperparameter set to use (e.g., 'flappybird1')", default="flappybird1")
    args = parser.parse_args()

    if args.algorithm == "dqn":
        dql = Agent(hyperparameter_set=args.hyperparameters)
        if args.train:
            dql.run(is_training=True, render=False)
        else:
            dql.run(is_training=False, render=True)
    elif args.algorithm == "a2c":
        if args.train:
            train_a2c()
        else:
            test_a2c()

# if __name__ == '__main__':
#     # Parse command line inputs
#     parser = argparse.ArgumentParser(description='Train or test model.')
#     parser.add_argument('hyperparameters', help='')
#     parser.add_argument('--train', help='Training mode', action='store_true')
#     args = parser.parse_args()

#     dql = Agent(hyperparameter_set=args.hyperparameters)

#     if args.train:
#         dql.run(is_training=True)
#     else:
#         dql.run(is_training=False, render=True)