import torch
from cartpole_gym import CustomCartPoleEnv
from model.amp import Policy


# --- Hyperparameters ---
POLICY_PATH = 'policy.pth'  # 학습된 policy 모델 경로
NUM_TEST_EPISODES = 5       # 테스트할 에피소드 수
MAX_STEPS = 1000            # 에피소드당 최대 스텝 (20초)
STATE_DIM = 4
ACTION_DIM = 1              # Continuous action: -1 ~ 1


def test_policy(policy_path, num_episodes=5, render=True):
    """학습된 policy를 테스트하고 시각화"""

    # 환경 초기화 (렌더링 활성화)
    render_mode = "human" if render else None
    env = CustomCartPoleEnv(render_mode=render_mode, max_episode_steps=MAX_STEPS)

    # Policy 로드
    policy = Policy(STATE_DIM, ACTION_DIM)
    try:
        policy.load_state_dict(torch.load(policy_path))
        policy.eval()
        print(f"✓ Policy 모델 로드 완료: {policy_path}\n")
    except FileNotFoundError:
        print(f"✗ 모델 파일을 찾을 수 없습니다: {policy_path}")
        print("먼저 train.py로 모델을 학습시켜주세요.\n")
        env.close()
        return

    total_rewards = []
    episode_lengths = []

    print(f"=== 테스트 시작: {num_episodes}개 에피소드 ===\n")

    for ep in range(num_episodes):
        state, _ = env.reset()
        state_t = torch.FloatTensor(state)

        episode_reward = 0.0
        steps = 0

        for _ in range(MAX_STEPS):
            # Policy로 행동 선택 (continuous action)
            with torch.no_grad():
                dist = policy(state_t.unsqueeze(0))
                action = dist.sample()  # 확률적 선택
                action = torch.clamp(action, -1.0, 1.0)  # 범위 제한
                # action = dist.mean  # 결정적 선택 (주석 해제하여 사용 가능)

            # 환경에서 실행
            action_np = action.squeeze().numpy()
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            next_state_t = torch.FloatTensor(next_state)

            episode_reward += reward
            steps += 1

            state_t = next_state_t

            # 에피소드 종료 체크
            if terminated or truncated:
                break

        total_rewards.append(episode_reward)
        episode_lengths.append(steps)

        print(f"Episode {ep+1}/{num_episodes} | "
              f"Steps: {steps:>3d} | "
              f"Reward: {episode_reward:>6.1f} | "
              f"Duration: {steps*0.02:.2f}s")

    env.close()

    # 통계 출력
    print(f"\n{'='*50}")
    print(f"테스트 결과 요약:")
    print(f"  평균 에피소드 길이: {sum(episode_lengths)/len(episode_lengths):.1f} steps")
    print(f"  평균 보상: {sum(total_rewards)/len(total_rewards):.1f}")
    print(f"  최대 보상: {max(total_rewards):.1f}")
    print(f"  최소 보상: {min(total_rewards):.1f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test trained AMP policy')
    parser.add_argument('--policy', type=str, default=POLICY_PATH,
                        help='Path to trained policy model')
    parser.add_argument('--episodes', type=int, default=NUM_TEST_EPISODES,
                        help='Number of test episodes')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering')

    args = parser.parse_args()

    test_policy(
        policy_path=args.policy,
        num_episodes=args.episodes,
        render=not args.no_render
    )
