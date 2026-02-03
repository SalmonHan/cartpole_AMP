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
def collect_trajectory(env, policy, max_steps, device=None):
    """현재 policy로 agent 궤적 수집 (task reward 포함, continuous action)"""
    if device is None:
        device = torch.device('cpu')

    states, actions, next_states, task_rewards = [], [], [], []
    state, _ = env.reset()
    state_t = torch.FloatTensor(state)

    for _ in range(max_steps):
        with torch.no_grad():
            # GPU에서 policy 추론
            dist = policy(state_t.unsqueeze(0).to(device))
            action = dist.sample()  # (1, action_dim) tensor
            action_clamped = torch.clamp(action, -1.0, 1.0)  # 범위 제한

        # 환경에 전달할 action (numpy array, CPU로 이동 필요)
        action_np = action_clamped.squeeze(0).cpu().numpy()
        next_state, reward, terminated, truncated, _ = env.step(action_np)
        next_state_t = torch.FloatTensor(next_state)

        states.append(state_t)
        actions.append(action_clamped.squeeze(0).cpu())  # CPU에서 저장
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


def collect_trajectory_vectorized(envs, policy, max_steps, device=None):
    """벡터화된 환경에서 병렬로 궤적 수집 (num_envs개 환경에서 동시 실행)"""
    if device is None:
        device = torch.device('cpu')

    num_envs = envs.num_envs
    states, actions, next_states, task_rewards = [], [], [], []

    # 모든 환경 초기화 (num_envs, state_dim)
    state, _ = envs.reset()
    state_t = torch.FloatTensor(state)  # (num_envs, state_dim)

    for _ in range(max_steps):
        with torch.no_grad():
            # 배치로 policy 추론 (num_envs, state_dim) -> (num_envs, action_dim)
            dist = policy(state_t.to(device))
            action = dist.sample()
            action_clamped = torch.clamp(action, -1.0, 1.0)

        # 환경에 전달할 action (numpy array)
        action_np = action_clamped.cpu().numpy()  # (num_envs, action_dim)
        next_state, reward, terminated, truncated, _ = envs.step(action_np)
        next_state_t = torch.FloatTensor(next_state)

        # 모든 환경의 데이터를 저장
        states.append(state_t)
        actions.append(action_clamped.cpu())
        next_states.append(next_state_t)
        task_rewards.append(torch.FloatTensor(reward))

        state_t = next_state_t
        # VectorEnv는 자동으로 terminated/truncated된 환경을 reset함

    # (max_steps, num_envs, dim) -> (max_steps * num_envs, dim)
    all_states = torch.stack(states)  # (max_steps, num_envs, state_dim)
    all_actions = torch.stack(actions)  # (max_steps, num_envs, action_dim)
    all_next_states = torch.stack(next_states)
    all_rewards = torch.stack(task_rewards)  # (max_steps, num_envs)

    # flatten: (max_steps * num_envs, dim)
    return (all_states.view(-1, all_states.shape[-1]),
            all_actions.view(-1, all_actions.shape[-1]),
            all_next_states.view(-1, all_next_states.shape[-1]),
            all_rewards.view(-1))


