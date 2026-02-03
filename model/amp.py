import torch
import torch.nn as nn


# --- Networks ---
class Discriminator(nn.Module):
    """(s, s') → P(expert)"""
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, s, s_next):
        x = torch.cat([s, s_next], dim=-1)
        return self.net(x).squeeze(-1)


class Policy(nn.Module):
    """s → action distribution (Gaussian for continuous action)"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # 학습 가능한 std

    def forward(self, s):
        features = self.net(s)
        mean = torch.tanh(self.mean_head(features))  # -1 ~ 1로 제한
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)


# --- Trajectory Collection ---
def collect_trajectory(env, policy, max_steps):
    """현재 policy로 agent 궤적 수집 (task reward 포함, continuous action)"""
    states, actions, next_states, task_rewards = [], [], [], []
    state, _ = env.reset()
    state_t = torch.FloatTensor(state)

    for _ in range(max_steps):
        with torch.no_grad():
            dist = policy(state_t.unsqueeze(0))
            action = dist.sample()  # (1, action_dim) tensor
            action_clamped = torch.clamp(action, -1.0, 1.0)  # 범위 제한

        # 환경에 전달할 action (numpy array)
        action_np = action_clamped.squeeze(0).numpy()
        next_state, reward, terminated, truncated, _ = env.step(action_np)
        next_state_t = torch.FloatTensor(next_state)

        states.append(state_t)
        actions.append(action_clamped.squeeze(0))  # (action_dim,) tensor
        next_states.append(next_state_t)
        task_rewards.append(reward)

        state_t = next_state_t
        if terminated or truncated:
            state, _ = env.reset()
            state_t = torch.FloatTensor(state)

    return (torch.stack(states),
            torch.stack(actions),  # (N, action_dim) tensor
            torch.stack(next_states),
            torch.tensor(task_rewards))


