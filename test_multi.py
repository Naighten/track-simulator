# test_model.py
import os
import time

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from CarGameEnv import CarGameEnv

# === ПАРАМЕТРЫ ===
TRACK_NAME = "track2"
MODEL_DIR = "./v1"
MODEL_NAME = "latest.zip"  # или best.zip, crash_save_*.zip
VECNORM_NAME = "vecnormalize.pkl"
RENDER_MODE = "human"
NUM_EPISODES = 5
STEP_DELAY = 0.01  # задержка между кадрами (сек)

# === УСТРОЙСТВО ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === ПОДГОТОВКА НОРМАЛИЗАТОРА ===
vecnorm_path = os.path.join(MODEL_DIR, VECNORM_NAME)
# Создаём векторную обёртку для нормализатора
dummy_env = DummyVecEnv([lambda: CarGameEnv(track_name=TRACK_NAME, render_mode=None)])
if os.path.exists(vecnorm_path):
    env_norm = VecNormalize(dummy_env, training=False, norm_reward=False)
    env_norm = VecNormalize.load(vecnorm_path, env_norm)
else:
    print(f"Warning: {VECNORM_NAME} not found, running without normalization.")
    env_norm = None

# === ЗАГРУЗКА МОДЕЛИ ===
model_path = os.path.join(MODEL_DIR, MODEL_NAME)
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Модель {model_path} не найдена!")
model = PPO.load(model_path, device=device)

# === ТЕСТИРОВАНИЕ ===
for ep in range(NUM_EPISODES):
    # Создаём среду для рендеринга
    raw_env = CarGameEnv(track_name=TRACK_NAME, render_mode=RENDER_MODE)
    obs_raw, _ = raw_env.reset()
    # Нормализуем obs, если есть нормализатор
    obs = env_norm.normalize_obs(obs_raw) if env_norm else obs_raw
    done = False
    total_reward = 0.0
    step = 0

    while not done:
        # Предсказание действия
        action, _ = model.predict(obs, deterministic=True)
        # Шаг в сырой среде
        obs_raw, reward, terminated, truncated, info = raw_env.step(int(action))
        # Подготовка obs для следующего шага
        obs = env_norm.normalize_obs(obs_raw) if env_norm else obs_raw
        total_reward += reward
        step += 1

        # Визуализация
        raw_env.render()
        time.sleep(STEP_DELAY)

        # Проверка завершения эпизода
        done = terminated or truncated

    print(
        f"Эпизод {ep + 1}: total_reward={total_reward:.2f}, steps={step}, success={info.get('success', False)}"
    )
    raw_env.close()
