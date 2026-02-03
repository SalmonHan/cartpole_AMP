import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class CustomCartPoleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, max_episode_steps=500):
        super(CustomCartPoleEnv, self).__init__()

        # 1. 물리 파라미터 정의
        self.M = 2.0
        self.m = 0.1
        self.l = 5.0
        self.g = 9.8
        self.dt = 0.001
        self.force_mag = 4.0


        # 초당 50번 업데이트 함. dt가 0.001이기 때문에 1step당 0.02초가 흐른다. 데이터를 20초동안 모았으면, 1000step 진행시키면 됨.
        self.render_fps = 50
        self.steps_per_render = int((1.0 / self.render_fps) / self.dt)


        # 최대 타임스텝 설정
        self.max_episode_steps = max_episode_steps

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)  # -1 ~ 1 연속 값

        high = np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.state = None
        
        # 렌더링 관련 변수
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        self.cart_rect = None
        self.pole_line = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # [추가됨] 현재 스텝 카운터 초기화
        self.current_step = 0

        initial_theta = np.random.uniform(-0.3, 0.3)
        self.state = np.array([0.0, 0.0, initial_theta, 0.0], dtype=np.float32)
        
        if self.render_mode == "human":
            self._render_frame()

        return self.state, {}

    def step(self, action):
        # [추가됨] 스텝 카운트 증가
        self.current_step += 1

        # Continuous action: -1 ~ 1 → -force_mag ~ +force_mag
        action_val = np.clip(action, -1.0, 1.0)
        if isinstance(action_val, np.ndarray):
            action_val = action_val[0]
        force = action_val * self.force_mag

        # 물리 시뮬레이션을 steps_per_render회 반복 (렌더링은 마지막에 한 번만)
        for _ in range(self.steps_per_render):
            x, x_dot, theta, theta_dot = self.state

            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)

            denom = self.l * (self.M + self.m * sin_theta**2)

            # [수정됨] centripetal 항의 부호: + → -
            numer_theta = (self.g * (self.M + self.m) * sin_theta - force * cos_theta -
                           self.m * self.l * theta_dot**2 * sin_theta * cos_theta)
            theta_two_dot = numer_theta / denom

            numer_x = ((force + self.m * self.l * theta_dot**2 * sin_theta) -
                       self.m * self.l * theta_two_dot * cos_theta)
            x_two_dot = numer_x / (self.M + self.m)

            x_dot = x_dot + x_two_dot * self.dt
            theta_dot = theta_dot + theta_two_dot * self.dt
            x = x + x_dot * self.dt
            theta = theta + theta_dot * self.dt

            self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        terminated = False

        # [수정됨] 시간 제한(Truncated) 조건
        truncated = (self.current_step >= self.max_episode_steps)

        # theta가 -pi/6 ~ pi/6, x가 -20 ~ 20 범위 내일 때 reward = 1
        x, theta = self.state[0], self.state[2]
        if -np.pi/6 <= theta <= np.pi/6 and -20 <= x <= 20:
            reward = 1.0
        else:
            reward = 0.0

        if self.render_mode == "human":
            self._render_frame()

        return self.state, reward, terminated, truncated, {}

    # render, _render_frame, close 메서드는 기존과 동일하므로 생략 가능
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        # (기존 코드와 동일)
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
            self.ax.set_xlim(-20, 20)
            self.ax.set_ylim(-20, 20)
            self.ax.grid(True)
            self.ax.set_xlabel('Position (m)')
            
            cart_width = 2.0
            cart_height = 1.0
            self.cart_rect = Rectangle((0, 0), cart_width, cart_height, facecolor='blue')
            self.ax.add_patch(self.cart_rect)
            
            self.pole_line, = self.ax.plot([], [], 'r', linewidth=3)
            self.ax.plot([-20, 20], [-cart_height/2, -cart_height/2], 'k', linewidth=2)

        x, _, theta, _ = self.state
        cart_width = 2.0
        cart_height = 1.0
        
        self.cart_rect.set_xy((x - cart_width/2, -cart_height/2))
        pole_x = x + 2 * self.l * np.sin(theta)
        pole_y = 2 * self.l * np.cos(theta)
        self.pole_line.set_data([x, pole_x], [0, pole_y])
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def close(self):
        if self.fig:
            plt.close(self.fig)

# === 사용 예시 ===
if __name__ == "__main__":
    env = CustomCartPoleEnv(render_mode="human")
    
    observation, info = env.reset()
    
    for _ in range(1000):
        action = env.action_space.sample() 
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        # truncated(시간 초과)가 True일 때 리셋됩니다.
        if terminated or truncated:
            print(f"에피소드 종료 (Step: {env.current_step}). 환경을 초기화합니다.")
            observation, info = env.reset()

    env.close()