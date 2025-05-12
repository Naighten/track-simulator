import json
import os

import cv2
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from CarGame import CarGame


class CarGameEnv(gym.Env):
    """Среда гонок с конфигурируемой системой вознаграждений"""

    metadata = {"render_modes": ["human"]}

    def __init__(
            self,
            track_name: str = "newtrack2",
            render_mode: str | None = "human",
            config_path: str = "tracks/tracks.json",
            seed: int = None,
            reward_config: dict | None = None,
    ):
        super().__init__()
        self.seed(seed) if seed else None

        # Загрузка конфигурации трека
        with open(config_path, "r") as f:
            data = json.load(f)
            track_cfg = next(
                (
                    entry[track_name]
                    for entry in data.get("tracks", [])
                    if track_name in entry
                ),
                None,
            )

        if not track_cfg:
            raise ValueError(f"Track '{track_name}' not found in {config_path}")

        # Инициализация игры
        self.game = CarGame(track_name, render_mode, seed=seed)
        self.render_mode = render_mode

        # Параметры вознаграждений
        self.default_reward_config = {
            # Базовые параметры
            "checkpoint": 50.0,  # Бонус за чекпоинт
            "lap_complete": 500.0,  # Бонус за полный круг
            "speed_coef": 2.0,  # Множитель награды за скорость
            "max_speed": self.game.max_speed,
            # Штрафы
            "offtrack_penalty": 2.0,  # Штраф за выезд с трассы
            "step_penalty": 0.05,  # Штраф за каждый шаг
            "reverse_penalty": 2.0,  # Штраф за движение назад
            "min_speed_penalty": 1.0,  # Штраф за низкую скорость
            "angle_penalty_coef": 0.05,  # Штраф за резкие повороты
            "proximity_coef": 0.5,  # Множитель награды за приближение к чекпоинту
            "max_proximity_bonus": 2.0,  # Максимальный бонус за шаг
            # Пороговые значения
            "checkpoint_threshold": 30.0,
            "angle_threshold": 35.0,  # Градусов на шаг
            "stuck_threshold": 500,  # Шагов без движения
            "min_speed": 0.1,  # Минимальная скорость
            # Параметры длительности эпизода
            "episode_settings": {
                "base_steps": 1500,
                "steps_per_checkpoint": 70,
                "max_extension": 3,
                "min_progress": 0.7,
            },
        }

        # Слияние переданного и дефолтного конфигов
        self.reward_config = self.default_reward_config.copy()
        if reward_config:
            self.reward_config.update(reward_config)
            if "episode_settings" in reward_config:
                self.reward_config["episode_settings"].update(
                    reward_config["episode_settings"]
                )

        # Инициализация трека
        img_path = os.path.join(
            os.path.dirname(config_path) or ".", track_cfg["track_image"]
        )
        self._init_track_mask(img_path)
        self.waypoints = np.array(
            [[wp["x"], wp["y"]] for wp in track_cfg["waypoints"]], dtype=np.float32
        )

        # Состояние среды
        self.next_wp_idx = 0
        self.passed_all_checkpoints = False
        self.is_invalid_lap = False
        self.prev_dist = None
        self.prev_angle = None
        self.cumulative_reward = 0.0
        self.step_count = 0

        # Пространства действий и наблюдений
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),  # [руль, газ]
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array(
                [0, 0, -360, -self.reward_config["max_speed"]] +
                [0, 0, -1, -1] + [0]*self.game.num_sensors,
                dtype=np.float32
            ),
            high=np.array(
                [
                    self.game.WIDTH,
                    self.game.HEIGHT,
                    360,
                    self.reward_config["max_speed"]
                ] +
                [1, 1, 1, 1] + [1]*self.game.num_sensors,
                dtype=np.float32
            ),
            dtype=np.float32,
        )

        self.prev_lap_count = self.game.laps
        self.max_episode_steps = self._calculate_max_steps()

    def _init_track_mask(self, img_path):
        """Инициализирует маску трека"""
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.track_mask = np.all(img_rgb == [180, 180, 180], axis=-1)
        self.game.track_mask = self.track_mask

    def update_rewards(self, new_config: dict):
        """Динамическое обновление параметров системы вознаграждений"""
        self.reward_config.update(new_config)

    def _calculate_max_steps(self):
        """Рассчитывает максимальную длительность эпизода"""
        cfg = self.reward_config["episode_settings"]
        return (
                cfg["base_steps"] + len(self.waypoints) * cfg["steps_per_checkpoint"]
        ) * cfg["max_extension"]

    def reset(self, seed=None, options=None):
        """Сброс среды в начальное состояние"""
        super().reset(seed=seed)
        self.game.reset()
        self.max_episode_steps = self._calculate_max_steps()
        self.prev_lap_count = self.game.laps

        # Сброс состояния
        self.next_wp_idx = 0
        self.passed_all_checkpoints = False
        self.is_invalid_lap = False
        self.cumulative_reward = 0.0
        self.step_count = 0

        # Инициализация позиции
        x, y = self.game.car_x, self.game.car_y
        current_target = self.waypoints[self.next_wp_idx]
        self.prev_dist = np.linalg.norm([x - current_target[0], y - current_target[1]])
        self.prev_angle = self.game.car_angle

        return self._get_obs(x, y), {}

    def step(self, action: np.ndarray):
        """Выполняет один шаг в среде"""
        self.step_count += 1

        # Динамическое продление эпизода
        if self._should_extend_episode():
            self.max_episode_steps += 200

        terminated = False
        truncated = False
        reward = -self.reward_config["step_penalty"]

        # Распаковываем и нормализуем действия
        steering_norm, throttle_norm = action

        # Конвертация в дискретные значения (для обратной совместимости)
        steering = int((steering_norm + 1) * 1)  # [-1,1] -> [0,2]
        throttle = int((throttle_norm + 1) * 1)  # [-1,1] -> [0,2]

        # Ограничиваем и преобразуем
        steering = np.clip(steering, 0, 2)
        throttle = np.clip(throttle, 0, 2)

        self.game.update_physics(steering, throttle)

        # Получаем новое состояние
        x, y = self.game.car_x, self.game.car_y
        angle = self.game.car_angle
        speed = self.game.car_speed
        obs = self._get_obs(x, y)

        # Расчет наград
        reward += self._calculate_speed_reward(speed)
        # reward += self._calculate_steering_penalty(angle)
        reward += self._calculate_checkpoint_reward(x, y)
        reward += self._calculate_track_penalty(x, y)

        # Проверка условий завершения
        terminated |= self._check_lap_completion()
        terminated |= self._check_out_of_bounds(x, y)
        terminated |= self._check_stuck(speed)
        truncated |= self.step_count >= self.max_episode_steps

        self.cumulative_reward += reward

        # Проверка прогресса
        if self._check_progress():
            truncated = True
            reward -= 10.0

        return obs, reward, terminated, truncated, self._get_info()

    def _should_extend_episode(self):
        """Проверка условий для продления эпизода"""
        progress = self.next_wp_idx / len(self.waypoints)
        remaining = (self.max_episode_steps - self.step_count) / self.max_episode_steps
        return progress > 0.7 and remaining < 0.3

    def _check_progress(self):
        """Проверка отсутствия прогресса"""
        min_progress = self.reward_config["episode_settings"]["min_progress"]
        if self.step_count > 0.5 * self.max_episode_steps:
            return self.next_wp_idx < len(self.waypoints) * min_progress
        return False

    def _get_obs(self, x, y):
        """Формирует наблюдение с сенсорами"""
        return np.array([
            x / self.game.WIDTH,
            y / self.game.HEIGHT,
            self.game.car_angle / 360.0,
            self.game.car_speed / self.game.max_speed,
            *self.game.sensor_data  # Добавляем данные сенсоров
        ], dtype=np.float32)

    def _calculate_speed_reward(self, speed):
        """Вычисляет награду/штраф за скорость"""
        reward = 0.0

        # Бонус за скорость
        speed_ratio = speed / self.reward_config["max_speed"]
        reward += np.clip(speed_ratio, 0, 1) * self.reward_config["speed_coef"]

        # Штрафы
        if speed <= 0:
            reward -= self.reward_config["reverse_penalty"]
        elif speed < self.reward_config["min_speed"]:
            reward -= self.reward_config["min_speed_penalty"]

        return reward

    def _calculate_steering_penalty(self, current_angle):
        """Штраф за резкие повороты"""
        delta_angle = abs(current_angle - self.prev_angle)
        self.prev_angle = current_angle

        if delta_angle > self.reward_config["angle_threshold"]:
            return -delta_angle * self.reward_config["angle_penalty_coef"]
        return 0.0

    def _calculate_checkpoint_reward(self, x, y):
        """Обработка чекпоинтов с наградой за приближение"""
        reward = 0.0
        current_target = self.waypoints[self.next_wp_idx % len(self.waypoints)]
        dist = np.linalg.norm([x - current_target[0], y - current_target[1]])

        # Награда за прогресс приближения
        if self.prev_dist is not None:
            delta_dist = self.prev_dist - dist  # Положительное значение = приближение
            proximity_reward = (
                    delta_dist
                    * self.reward_config["proximity_coef"]
                    * (self.next_wp_idx / len(self.waypoints))
            )

            # Ограничиваем максимальный бонус/штраф
            proximity_reward = np.clip(
                proximity_reward,
                -self.reward_config["max_proximity_bonus"],
                self.reward_config["max_proximity_bonus"],
            )
            reward += proximity_reward if not self.is_invalid_lap else 0

        # Обновляем предыдущее расстояние ДО проверки достижения чекпоинта
        self.prev_dist = dist

        # Награда за достижение чекпоинта
        if dist < self.reward_config["checkpoint_threshold"]:
            self.next_wp_idx += 1
            if self.next_wp_idx >= len(self.waypoints):
                self.passed_all_checkpoints = True
            reward += self.reward_config["checkpoint"]

            # Сброс расстояния для нового чекпоинта
            if self.next_wp_idx < len(self.waypoints):
                next_target = self.waypoints[self.next_wp_idx]
                self.prev_dist = np.linalg.norm(
                    [x - next_target[0], y - next_target[1]]
                )

        return reward

    def _calculate_track_penalty(self, x, y):
        """Проверка нахождения на трассе"""
        xi = int(np.clip(x, 0, self.track_mask.shape[1] - 1))
        yi = int(np.clip(y, 0, self.track_mask.shape[0] - 1))

        if not self.track_mask[yi, xi]:
            self.is_invalid_lap = True
            return -self.reward_config["offtrack_penalty"]
        else:
            return self.reward_config["offtrack_penalty"]

    def _check_lap_completion(self):
        """Проверка завершения круга"""
        if self.game.laps >= 1:
            self.cumulative_reward += self.reward_config["lap_complete"]
            self.prev_lap_count = self.game.laps
            self.next_wp_idx = 0
            # if self.passed_all_checkpoints and not self.is_invalid_lap:
            return True
        return False

    def _check_out_of_bounds(self, x, y):
        """Проверка выхода за границы"""
        return not (1 <= x <= self.game.WIDTH - 1 and 1 <= y <= self.game.HEIGHT - 1)

    def _check_stuck(self, speed):
        """Проверка застревания"""
        return (
                speed < self.reward_config["min_speed"]
                and self.step_count > self.reward_config["stuck_threshold"]
        )

    def _get_info(self):
        """Дополнительная информация"""
        return {
            "checkpoints": self.next_wp_idx,
            "valid_lap": self.passed_all_checkpoints and not self.is_invalid_lap,
            "total_reward": self.cumulative_reward,
        }

    def render(self, mode="human"):
        """Отрисовка среды"""
        self.game.render()
        if mode == "human" and self.game.screen:
            self._render_hud()

    def _render_hud(self):
        """Отрисовка HUD"""
        font = pygame.font.SysFont(None, 24)
        texts = [
            f"Checkpoints: {self.next_wp_idx}/{len(self.waypoints)}",
            f"Reward: {self.cumulative_reward:.1f}",
            f"Steps: {self.step_count}",
        ]

        y_pos = 80
        for text in texts:
            surface = font.render(text, True, self.game.WHITE)
            self.game.screen.blit(surface, (20, y_pos))
            y_pos += 30

        pygame.display.flip()

    def close(self):
        """Завершение работы среды"""
        self.game.close()
