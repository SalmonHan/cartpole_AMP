import torch
import torch.optim as optim
import scipy.io as sio
import os
import glob
from gymnasium.vector import SyncVectorEnv
from cartpole_gym import CustomCartPoleEnv
from model.amp import Discriminator, Policy, collect_trajectory_vectorized


# --- Device 설정 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Hyperparameters ---
EXPERT_DATA_FOLDER = 'expert_data'  # 폴더 경로로 변경
NUM_EPISODES = 500                   # 학습 몇번 할건지
MAX_STEPS = 1000                   # 환경당 스텝 수
NUM_ENVS = 10                      # 병렬 환경 개수
BATCH_SIZE = 256
DISC_LR = 1e-3
POLICY_LR = 1e-4
GAMMA = 0.99
STATE_DIM = 4
ACTION_DIM = 1  # Continuous action: -1 ~ 1
STYLE_REWARD_WEIGHT = 0.5  # Style reward (discriminator) 가중치
TASK_REWARD_WEIGHT = 0.5   # Task reward (환경) 가중치


# --- Data Loading ---
def load_expert_data(folder_path):
    #폴더 내 모든 .mat 파일의 state들로 (s, s') 쌍 생성
    all_s = []
    all_s_next = []

    # 폴더 내 모든 .mat 파일 찾기
    mat_files = glob.glob(os.path.join(folder_path, '*.mat'))

    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {folder_path}")

    print(f"Found {len(mat_files)} .mat file(s) in {folder_path}")

    # 각 파일에서 데이터 로드
    for file_path in mat_files:
        print(f"  Loading: {os.path.basename(file_path)}")
        data = sio.loadmat(file_path)
        states = torch.FloatTensor(data['states'])  # (N, 4)

        s = states[:-1]       # (N-1, 4)
        s_next = states[1:]   # (N-1, 4)

        all_s.append(s)
        all_s_next.append(s_next)
        print(f"    Loaded {s.shape[0]} (s, s') pairs")

    # 모든 데이터 결합
    expert_s = torch.cat(all_s, dim=0)
    expert_s_next = torch.cat(all_s_next, dim=0)

    return expert_s, expert_s_next


# --- Training Loop ---
def train():
    # Expert data 로드 및 GPU로 이동
    expert_s, expert_s_next = load_expert_data(EXPERT_DATA_FOLDER)
    expert_s = expert_s.to(device)
    expert_s_next = expert_s_next.to(device)
    print(f"\nTotal expert data: {expert_s.shape[0]} (s, s') pairs\n")

    # 벡터 환경 초기화 (10개 환경 병렬 실행)
    envs = SyncVectorEnv([
        lambda: CustomCartPoleEnv(max_episode_steps=MAX_STEPS)
        for _ in range(NUM_ENVS)
    ])
    disc = Discriminator(STATE_DIM).to(device)
    policy = Policy(STATE_DIM, ACTION_DIM).to(device)

    disc_opt = optim.Adam(disc.parameters(), lr=DISC_LR)
    policy_opt = optim.Adam(policy.parameters(), lr=POLICY_LR)

    for ep in range(NUM_EPISODES):
        # 전체 학습 과정 : 
        # agent 궤적 수집하여 replay buffer에 저장한다.
        # 한 에피소드에 1000스텝이니까 10000번정돈 해야겠지?
        # 전부 replay buffer에 저장.

        # 그러고 나면 이제, 전문가에서 batch만큼 꺼내오고, replay buffer에서도 batch만큼 꺼내와서, 판별자 학습시키고,
        # task reward랑 판별자 reward 두개 사용해서 policy 학습

        # 이하 반복.


        # 1. Agent 궤적 수집해서 replay buffer에 저장 (GPU로 이동)
        # 10개 환경에서 각 1000 step = 총 10000 step 수집
        agent_s, agent_a, agent_s_next, task_rewards = collect_trajectory_vectorized(envs, policy, MAX_STEPS, device)
        agent_s = agent_s.to(device)
        agent_a = agent_a.to(device)
        agent_s_next = agent_s_next.to(device)
        task_rewards = task_rewards.to(device)
    


        # 2. Discriminator 학습
        # Expert → D 출력 1, Agent → D 출력 0
        expert_idx = torch.randint(0, expert_s.shape[0], (BATCH_SIZE,), device=device)
        agent_idx = torch.randint(0, agent_s.shape[0], (BATCH_SIZE,), device=device)

        expert_out = disc(expert_s[expert_idx], expert_s_next[expert_idx])
        agent_out = disc(agent_s[agent_idx], agent_s_next[agent_idx])

        disc_loss = -torch.mean(
            torch.log(expert_out + 1e-8) +
            torch.log(1 - agent_out + 1e-8)
        )

        disc_opt.zero_grad()
        disc_loss.backward()
        disc_opt.step()

        # 3. Policy 학습 (AMP)
        # Total Reward = Style Reward + Task Reward
        # 강화학습 알고리즘은 그냥 간단하게 policy gradient 사용(advantage function사용)
        with torch.no_grad():
            # Style reward: log D(s, s') - expert 동작 모방
            style_rewards = torch.log(disc(agent_s, agent_s_next) + 1e-8)

            # Task reward: 환경에서 받은 실제 보상
            # task_rewards는 이미 tensor로 반환됨

            # AMP 총 보상 = 가중 평균
            rewards = STYLE_REWARD_WEIGHT * style_rewards + TASK_REWARD_WEIGHT * task_rewards

        # Discounted returns 계산 (GPU에서 처리, .item() 제거로 동기화 방지)
        returns = torch.zeros_like(rewards)
        running_return = torch.tensor(0.0, device=device)
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + GAMMA * running_return
            returns[t] = running_return

        # Baseline (advantage function)
        returns = returns - returns.mean()

        # log π(a|s) 배치 계산 (for 루프 제거 → 대폭 속도 향상)
        dist = policy(agent_s)  # 전체 배치 한번에 처리
        log_probs = dist.log_prob(agent_a).sum(-1)

        policy_loss = -torch.mean(log_probs * returns)

        policy_opt.zero_grad()
        policy_loss.backward()
        policy_opt.step()

        # Logging
        if (ep + 1) % 10 == 0:
            print(f"Ep {ep+1:>4d}/{NUM_EPISODES} | "
                  f"Disc: {disc_loss.item():.4f} | "
                  f"Policy: {policy_loss.item():.4f} | "
                  f"Traj Len: {len(agent_s):>4d} | "
                  f"Style R: {style_rewards.mean().item():+.4f} | "
                  f"Task R: {task_rewards.mean().item():+.4f} | "
                  f"Total R: {rewards.mean().item():+.4f}")

    envs.close()

    # 학습된 모델 저장
    torch.save(policy.state_dict(), 'policy.pth')
    torch.save(disc.state_dict(), 'discriminator.pth')
    print("\n✓ 모델 저장 완료:")
    print(f"  - policy.pth")
    print(f"  - discriminator.pth")


if __name__ == "__main__":
    train()
