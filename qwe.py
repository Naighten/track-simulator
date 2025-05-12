import pygame

from CarGameEnv import CarGameEnv  # Проверьте путь, если нужно


def main():
    pygame.init()

    # Создаём среду
    env = CarGameEnv(track_name="track2", render_mode="human")

    # Таймер для ограничения FPS
    clock = pygame.time.Clock()

    # Сброс среды
    obs, _ = env.reset()

    running = True
    while running:
        # Ограничиваем частоту кадров
        clock.tick(env.game.FPS)

        # Обработка системных событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Чтение клавиш
        keys = pygame.key.get_pressed()

        # Определяем throttle: 0=reverse,1=idle,2=forward
        if keys[pygame.K_UP]:
            throttle = 2
        elif keys[pygame.K_DOWN]:
            throttle = 0
        else:
            throttle = 1

        # Определяем steering: 0=left,1=straight,2=right
        if keys[pygame.K_LEFT]:
            steering = 0
        elif keys[pygame.K_RIGHT]:
            steering = 2
        else:
            steering = 1

        # Собираем единое действие
        action = throttle * 3 + steering

        # Делает шаг в среде
        obs, reward, terminated, truncated, info = env.step(action)

        # Отрисовываем
        env.render()

        # Если эпизод завершён — выходим
        if terminated or truncated:
            print(
                f"Эпизод завершён, reward={reward:.2f}, success={info.get('success'), info}"
            )
            running = False

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
