import sys

import pygame
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from CarGameEnv import CarGameEnv  # Ваша среда


def train_and_demo(
    track_name: str = "track2",
    iterations: int = 100,
    timesteps_per_iter: int = 1000,
    pause_key: int = pygame.K_SPACE,
):
    # Инициализация Pygame для ожидания событий
    pygame.init()

    # Выбираем устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Создаём векторизованную среду для обучения ---
    train_env = DummyVecEnv([lambda: CarGameEnv(track_name=track_name)])

    # Инициализация модели PPO
    model = PPO("MlpPolicy", train_env, verbose=1, device=device)

    # --- Основной цикл обучения с демонстрацией ---
    for i in range(1, iterations + 1):
        # Обучаем модель на указанные timesteps
        model.learn(total_timesteps=timesteps_per_iter)
        print(f"\n=== Итерация {i}/{iterations} завершена ===")

        # --- Демонстрация на чистой среде с визуализацией ---
        demo_env = CarGameEnv(track_name=track_name)
        obs = demo_env.reset()
        done = False
        while not done:
            # Предсказание и шаг
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = demo_env.step(action)
            # Обработка событий и отрисовка
            pygame.event.pump()
            demo_env.render()
            # Лёгкая задержка, чтобы не гонять на 100% CPU
            pygame.time.delay(10)

        # Ожидаем, пока пользователь не нажмёт указанную клавишу
        print(
            f"Демонстрация завершена. Нажмите '{pygame.key.name(pause_key)}', чтобы продолжить..."
        )
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pause_key:
                    waiting = False
                elif event.type == pygame.QUIT:
                    demo_env.close()
                    sys.exit()
            pygame.time.delay(50)
        # Закрываем демонстрационное окно перед следующей итерацией
        demo_env.close()

    # Сохраняем модель после всех итераций
    model.save("ppo_car_game")
    print("Обучение завершено, модель сохранена как 'ppo_car_game'.")


if __name__ == "__main__":
    # Параметры можно менять по вкусу
    train_and_demo(
        track_name="track2",
        iterations=100,
        timesteps_per_iter=1000,
        pause_key=pygame.K_SPACE,
    )
