import numpy as np
import pygame

from CarGameEnv import CarGameEnv


def main():
    pygame.init()

    # Создаём среду с новым форматом действий
    env = CarGameEnv(track_name="newtrack2", render_mode="human")

    clock = pygame.time.Clock()
    obs, _ = env.reset()

    running = True
    while running:
        clock.tick(env.game.FPS)

        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Чтение клавиш
        keys = pygame.key.get_pressed()

        # Непрерывное управление
        steering = 0.0
        throttle = 0.0

        # Управление рулём
        if keys[pygame.K_LEFT]:
            steering = -1.0
        elif keys[pygame.K_RIGHT]:
            steering = 1.0

        # Управление газом/тормозом
        if keys[pygame.K_UP]:
            throttle = 1.0
        elif keys[pygame.K_DOWN]:
            throttle = -1.0

        # Формируем действие в новом формате
        action = np.array([steering, throttle], dtype=np.float32)

        # Шаг в среде
        obs, reward, terminated, truncated, info = env.step(action)

        # Отрисовка
        env.render()

        if terminated or truncated:
            print(f"Эпизод завершён! Награда: {info}")
            running = False

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
