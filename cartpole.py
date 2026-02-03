import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.io as sio
import time
import os
from datetime import datetime
import pygame


# 데이터 수집할 곳임.
def cartpole_optimized():
    # 물리 파라미터
    M = 2.0
    m = 0.1
    l = 5.0
    g = 9.8
    dt = 0.001           # 물리 시뮬레이션 간격 (1ms)
    force_mag = 4.0

    # 상태 벡터: [x, x_dot, theta, theta_dot]
    initial_theta = np.random.uniform(-np.pi / 4, np.pi / 4)
    state = np.array([0.0, 0.0, initial_theta, 0.0])

    # 데이터 저장 관련
    max_duration = 10.0
    elapsed_time = 0.0
    state_history = []   # gym step마다 state 저장
    action_history = []  # gym step마다 discrete action 저장
    is_saved = False

    # 시각화 설정 (Interactive 모드 활성화)
    plt.ion() 
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.canvas.manager.set_window_title('Optimized Cart-Pole')
    
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.grid(True)
    ax.set_xlabel('Position (m)')
    
    # 그래픽 객체
    cart_width = 2.0
    cart_height = 1.0
    cart_rect = Rectangle((state[0] - cart_width/2, -cart_height/2), 
                          cart_width, cart_height, facecolor='blue')
    ax.add_patch(cart_rect)
    
    pole_line, = ax.plot([], [], 'r', linewidth=3) # 초기 데이터 비움
    ax.plot([-20, 20], [-cart_height/2, -cart_height/2], 'k', linewidth=2)

    # 플레이스테이션 컨트롤러 초기화
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("컨트롤러가 연결되지 않았습니다!")
        return

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"컨트롤러 연결됨: {joystick.get_name()}")

   

    target_fps = 50                 # 목표 프레임 레이트 (초당 50회 그리기)
    render_interval = 1.0 / target_fps 
    steps_per_frame = int(render_interval / dt) # 한 번 그릴 때 수행할 물리 연산 횟수 (20회 : gym쪽 설정이랑 동일)
    
    start_time = time.time() # 실제 시간 측정용

    while plt.fignum_exists(fig.number):

        if elapsed_time > max_duration:
            if not is_saved:
                # expert_data 폴더 생성 (없으면)
                save_folder = 'expert_data'
                os.makedirs(save_folder, exist_ok=True)

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = os.path.join(save_folder, f'expert_data_{timestamp}.mat')
                sio.savemat(filename, {
                    'states': np.array(state_history),   # (N, 4)
                })
                print(f'{filename} 저장 완료. 총 {len(state_history)} steps')
                is_saved = True
            plt.pause(0.1)
            break

        # 조이스틱 입력 읽기
        pygame.event.pump()
        left_x = joystick.get_axis(0)  # -1(왼쪽) ~ +1(오른쪽) get_axis(0)이 x축이라는 뜻
        f = left_x * force_mag  # -force_mag ~ +force_mag로 매핑


        # 기록: gym step 한 회당 (state, action) 한 쌍
        state_history.append(state.copy())


        # 물리 연산을 steps_per_frame회 반복
        for _ in range(steps_per_frame):
            theta, theta_dot = state[2], state[3]

            denom = l * (M + m * np.sin(theta)**2)
            numer_theta = (g * (M + m) * np.sin(theta) - f * np.cos(theta) -
                           m * l * theta_dot**2 * np.sin(theta) * np.cos(theta))
            theta_two_dot = numer_theta / denom

            numer_x = ((f + m * l * theta_dot**2 * np.sin(theta)) -
                       (m * l * theta_two_dot * np.cos(theta)))
            x_two_dot = numer_x / (M + m)

            state[1] += x_two_dot * dt
            state[3] += theta_two_dot * dt
            state[0] += state[1] * dt
            state[2] += state[3] * dt

            elapsed_time += dt

        # 화면 갱신
        cart_rect.set_x(state[0] - cart_width/2)
        pole_x = state[0] + 2 * l * np.sin(state[2])
        pole_y = 2 * l * np.cos(state[2])
        pole_line.set_data([state[0], pole_x], [0, pole_y])

        plt.pause(0.001)

    pygame.quit()

if __name__ == "__main__":
    cartpole_optimized()