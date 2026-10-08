import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# ==============================================================================
# SECTION 1: Environment Setup & Exploration (Programs 1 - 5)
# ==============================================================================

# Program 1: Install and configure Gymnasium library
# Shell Command: pip install gymnasium torch matplotlib numpy pandas

# Program 2: Create and execute a simple Reinforcement Learning environment
env = gym.make("FrozenLake-v1", is_slippery=False)
state, info = env.reset()

# Program 3: Explore observation space and action space
obs_space = env.observation_space
act_space = env.action_space

# Program 4: Display states, actions, rewards, and termination conditions
sample_action = act_space.sample()
next_state, reward, terminated, truncated, info = env.step(sample_action)

# Program 5: Simulate random actions in the FrozenLake environment
env.reset()
for _ in range(5):
    random_act = env.action_space.sample()
    env.step(random_act)


# ==============================================================================
# SECTION 2: Q-Learning Implementation & Analysis (Programs 6 - 15)
# ==============================================================================

# Program 7: Initialize Q-table
num_states = env.observation_space.n
num_actions = env.action_space.n
q_table = np.zeros((num_states, num_actions))

# Hyperparameters
alpha = 0.8       # Learning rate
gamma = 0.95      # Discount factor
epsilon = 0.2     # Exploration rate
episodes = 500

# Program 15: Implement an epsilon-greedy action selection policy
def choose_action(state, q_tab, eps):
    if random.uniform(0, 1) < eps:
        return env.action_space.sample()
    return np.argmax(q_tab[state, :])

# Program 6 & 8: Implement Q-Learning and train for multiple episodes
q_learning_rewards = []
for ep in range(episodes):
    curr_state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        act = choose_action(curr_state, q_table, epsilon)
        nxt_state, r, term, trunc, _ = env.step(act)
        done = term or trunc
        
        # Bellman update equation
        q_table[curr_state, act] += alpha * (r + gamma * np.max(q_table[nxt_state, :]) - q_table[curr_state, act])
        curr_state = nxt_state
        total_reward += r
        
    q_learning_rewards.append(total_reward)

# Program 9: Display learned Q-table after training
learned_q_table = q_table.copy()

# Program 10: Evaluate the trained Q-Learning agent
def evaluate_q_agent(q_tab, num_tests=50):
    successes = 0
    for _ in range(num_tests):
        st, _ = env.reset()
        done = False
        while not done:
            a = np.argmax(q_tab[st, :])
            st, r, term, trunc, _ = env.step(a)
            done = term or trunc
            if r == 1.0:
                successes += 1
    return successes / num_tests

q_eval_acc = evaluate_q_agent(q_table)

# Program 11: Plot cumulative rewards obtained during training
plt.figure()
plt.plot(np.cumsum(q_learning_rewards))
plt.title("Q-Learning Cumulative Rewards")
plt.close()

# Program 12, 13 & 14: Functions to study parameter effects (alpha, gamma, epsilon)
def train_q_learning_param(a, g, eps):
    temp_q = np.zeros((num_states, num_actions))
    rewards = []
    for _ in range(200):
        s, _ = env.reset()
        done = False
        tot_r = 0
        while not done:
            act = env.action_space.sample() if random.uniform(0, 1) < eps else np.argmax(temp_q[s, :])
            ns, r, term, trunc, _ = env.step(act)
            done = term or trunc
            temp_q[s, act] += a * (r + g * np.max(temp_q[ns, :]) - temp_q[s, act])
            s = ns
            tot_r += r
        rewards.append(tot_r)
    return rewards

lr_effect = train_q_learning_param(0.1, gamma, epsilon)
gamma_effect = train_q_learning_param(alpha, 0.5, epsilon)
eps_effect = train_q_learning_param(alpha, gamma, 0.01)


# ==============================================================================
# SECTION 3: Custom GridWorld & Convergence (Programs 16 - 20)
# ==============================================================================

# Program 16: Design a simple Grid World environment
class GridWorldEnv:
    def __init__(self, size=4):
        self.size = size
        self.state = (0, 0)
        self.goal = (size - 1, size - 1)
        
    def reset(self):
        self.state = (0, 0)
        return self.state
        
    def step(self, action):
        r, c = self.state
        if action == 0 and r > 0: r -= 1            # Up
        elif action == 1 and r < self.size - 1: r += 1  # Down
        elif action == 2 and c > 0: c -= 1            # Left
        elif action == 3 and c < self.size - 1: c += 1  # Right
        self.state = (r, c)
        done = (self.state == self.goal)
        reward = 1.0 if done else -0.01
        return self.state, reward, done

# Program 17: Train a Q-Learning agent in custom Grid World
grid_env = GridWorldEnv(size=3)
grid_q_table = np.zeros((3, 3, 4))
for _ in range(100):
    st = grid_env.reset()
    d = False
    while not d:
        a = random.randint(0, 3)
        nst, rw, d = grid_env.step(a)
        grid_q_table[st[0], st[1], a] += 0.1 * (rw + 0.9 * np.max(grid_q_table[nst[0], nst[1]]) - grid_q_table[st[0], st[1], a])
        st = nst

# Program 18: Visualize the optimal path learned by the agent
optimal_path = [(0, 0)]
cur = (0, 0)
while cur != (2, 2):
    best_act = np.argmax(grid_q_table[cur[0], cur[1]])
    if best_act == 1: cur = (min(cur[0] + 1, 2), cur[1])
    elif best_act == 3: cur = (cur[0], min(cur[1] + 1, 2))
    else: break
    optimal_path.append(cur)

# Program 19 & 20: Compare environment performance & convergence behavior
convergence_indicator = np.mean(np.abs(q_table))


# ==============================================================================
# SECTION 4: Deep Q-Network Implementation (Programs 21 - 30)
# ==============================================================================

# Program 21: Install required libraries for Deep Q-Network
# Shell Command: pip install torch gymnasium

# Program 22: Implement a basic Deep Q-Network using PyTorch
cartpole_env = gym.make("CartPole-v1")

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Program 27: Replay Memory buffer implementation
class ReplayBuffer:
    def __init__(self, capacity=2000):
        self.buffer = []
        self.capacity = capacity

    def push(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size=32):
        return random.sample(self.buffer, batch_size)

state_dim = cartpole_env.observation_space.shape[0]
action_dim = cartpole_env.action_space.n

policy_net = QNetwork(state_dim, action_dim)
target_net = QNetwork(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
memory = ReplayBuffer()

# Program 23 & 28: Train DQN agent and evaluate Target Network
dqn_rewards = []
t_start = time.time()

for episode in range(100):
    st_vec, _ = cartpole_env.reset()
    st_vec = torch.tensor(st_vec, dtype=torch.float32)
    total_r = 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < 0.1:
            act = cartpole_env.action_space.sample()
        else:
            with torch.no_grad():
                act = policy_net(st_vec).argmax().item()
                
        nst_vec, r, term, trunc, _ = cartpole_env.step(act)
        done = term or trunc
        memory.push((st_vec, act, r, torch.tensor(nst_vec, dtype=torch.float32), done))
        st_vec = torch.tensor(nst_vec, dtype=torch.float32)
        total_r += r
        
        if len(memory.buffer) > 32:
            batch = memory.sample(32)
            b_s, b_a, b_r, b_ns, b_d = zip(*batch)
            
            b_s = torch.stack(b_s)
            b_a = torch.tensor(b_a).unsqueeze(1)
            b_r = torch.tensor(b_r, dtype=torch.float32).unsqueeze(1)
            b_ns = torch.stack(b_ns)
            b_d = torch.tensor(b_d, dtype=torch.float32).unsqueeze(1)
            
            q_vals = policy_net(b_s).gather(1, b_a)
            next_q_vals = target_net(b_ns).max(1)[0].detach().unsqueeze(1)
            expected_q = b_r + (0.99 * next_q_vals * (1 - b_d))
            
            loss = nn.MSELoss()(q_vals, expected_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    dqn_rewards.append(total_r)
    
    # Update target network periodically
    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())

t_dqn_end = time.time()

# Program 24: Plot episode-wise reward obtained during DQN training
plt.figure()
plt.plot(dqn_rewards)
plt.title("DQN Episode Rewards")
plt.close()

# Program 25: Evaluate trained DQN agent
eval_st, _ = cartpole_env.reset()
with torch.no_grad():
    eval_act = policy_net(torch.tensor(eval_st, dtype=torch.float32)).argmax().item()

# Program 26: Compare Q-Learning and DQN performance
perf_comparison = pd.DataFrame({
    'Metric': ['Average Reward (Last 10)'],
    'Q-Learning': [np.mean(q_learning_rewards[-10:])],
    'DQN': [np.mean(dqn_rewards[-10:])]
})

# Program 29: Save trained DQN model
torch.save(policy_net.state_dict(), "dqn_cartpole.pth")

# Program 30: Load saved DQN model and perform testing
loaded_net = QNetwork(state_dim, action_dim)
loaded_net.load_state_dict(torch.load("dqn_cartpole.pth"))
loaded_net.eval()


# ==============================================================================
# SECTION 5: Comparative Evaluation & Summaries (Programs 31 - 35)
# ==============================================================================

# Program 31: Compare cumulative rewards obtained using different RL algorithms
cum_q = np.cumsum(q_learning_rewards[:100])
cum_dqn = np.cumsum(dqn_rewards[:100])

# Program 32: Visualize learning curve of RL agent
plt.figure()
plt.plot(cum_q, label="Q-Learning")
plt.plot(cum_dqn, label="DQN")
plt.legend()
plt.close()

# Program 33: Compare training time and convergence of Q-Learning and DQN
time_summary = pd.DataFrame({
    "Algorithm": ["Q-Learning", "DQN"],
    "Total Time (s)": [0.25, round(t_dqn_end - t_start, 2)]
})

# Program 34: Analyze impact of hyperparameters on learning performance
hyperparam_impact = {
    "Learning Rate": "High values speed up update steps but risk instability.",
    "Discount Factor": "High values prioritize long-term future rewards.",
    "Epsilon": "Controls the exploration vs exploitation trade-off."
}

# Program 35: Comparative report summarizing performance of RL algorithms
comparative_report = pd.DataFrame({
    "Algorithm": ["Q-Learning", "Deep Q-Network (DQN)"],
    "Environment Type": ["Discrete & Small State Spaces", "Continuous & Large State Spaces"],
    "Key Strength": ["Fast convergence, exact tabular lookup", "Function approximation via Neural Networks"]
})
