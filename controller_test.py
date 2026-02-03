import pygame
import sys

def main():
    pygame.init()
    pygame.joystick.init()

    # 연결된 조이스틱 확인
    joystick_count = pygame.joystick.get_count()
    if joystick_count == 0:
        print("컨트롤러가 연결되지 않았습니다.")
        sys.exit(1)

    # 첫 번째 조이스틱 사용
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    print(f"컨트롤러 이름: {joystick.get_name()}")
    print(f"축 개수: {joystick.get_numaxes()}")
    print(f"버튼 개수: {joystick.get_numbuttons()}")
    print("\n왼쪽 조이스틱 값을 표시합니다. (Ctrl+C로 종료)\n")

    clock = pygame.time.Clock()

    try:
        while True:
            pygame.event.pump()

            # 왼쪽 조이스틱: 보통 축 0(X), 축 1(Y)
            left_x = joystick.get_axis(0)  # 좌우 (-1: 왼쪽, 1: 오른쪽)
            left_y = joystick.get_axis(1)  # 상하 (-1: 위, 1: 아래)

            # 시각화를 위한 바 생성
            bar_length = 20
            x_pos = int((left_x + 1) / 2 * bar_length)
            y_pos = int((left_y + 1) / 2 * bar_length)

            x_bar = "[" + "=" * x_pos + "|" + "=" * (bar_length - x_pos) + "]"
            y_bar = "[" + "=" * y_pos + "|" + "=" * (bar_length - y_pos) + "]"

            print(f"\rX: {left_x:+.3f} {x_bar}  Y: {left_y:+.3f} {y_bar}    ", end="")

            clock.tick(30)  # 30 FPS

    except KeyboardInterrupt:
        print("\n\n종료합니다.")
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()
