import json
import math
import random
import time

import pygame

from config import get_config, Config


class CarGame:
    _config: Config = get_config()

    def __init__(self, track_name="newtrack2", render_mode="human", seed=None):
        self.render_mode = render_mode

        # Инициализация генератора случайных чисел для воспроизводимости
        if seed is not None:
            random.seed(seed)
            # Если используете numpy где-то:
            import numpy as np

            np.random.seed(seed)

        self.FPS = self._config.FPS
        self.WIDTH, self.HEIGHT = self._config.WIDTH, self._config.HEIGHT

        # Цвета и графика
        self.GREY = self._config.GREY
        self.TRACK_COLOR = self._config.TRACK_COLOR
        self.WHITE = self._config.WHITE
        self.RED = self._config.RED
        self.car_width = self._config.CAR_WIDTH
        self.car_height = self._config.CAR_HEIGHT
        self.car_scale = self._config.CAR_SCALE

        # Параметры машины
        self.car_speed = 0
        self.max_speed = self._config.MAX_SPEED
        self.acceleration = self._config.ACCELERATION
        self.rotation_speed = self._config.ROTATION_SPEED
        self.drift_factor = self._config.DRIFT_FACTOR

        # Для маски трассы (если передана из внешнего модуля)
        self.track_mask = None

        # Данные трассы
        self.track_data = self.load_track_data("tracks/tracks.json", track_name)
        if self.track_data is None:
            raise ValueError(f"Track with name '{track_name}' not found.")

        match self.render_mode:
            case "human":
                pygame.init()
                self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
                pygame.display.set_caption("Машинка с дрифтом")
                self.font = pygame.font.SysFont(None, 24)

                self.default_image = pygame.image.load("car.png").convert_alpha()
                self.dead_image = pygame.image.load("red_car.png").convert_alpha()

                self.track = pygame.image.load("tracks/" + self.track_data["track_image"]).convert()
            case _:
                self.screen = None
                self.font = None

                self.default_image = None
                self.dead_image = None

                self.track = None

        self.car_image = self.default_image

        # Координаты финишной линии
        self.A = self.track_data["finish_line"]["A"]
        self.B = self.track_data["finish_line"]["B"]
        self.C = self.track_data["finish_line"]["C"]
        self.start_coords = {
            "x_min": self.track_data["start_coords"]["x_min"],
            "x_max": self.track_data["start_coords"]["x_max"],
            "y_min": self.track_data["start_coords"]["y_min"],
            "y_max": self.track_data["start_coords"]["y_max"],
        }

        # Установка начального положения и ориентации машины
        self.car_x = self.track_data["start_position"]["x"]
        self.car_y = self.track_data["start_position"]["y"]
        self.car_angle = -math.degrees(math.atan(-(self.A / self.B)))
        self.velocity_angle = self.car_angle

        # Настройки для гонки
        self.laps = -1
        self.start_time = None
        self.is_invalid_lap = False
        self.previous_f_result = None
        self.best_lap_time = None
        self.last_valid_lap_time = None

        # Параметры сенсоров
        self.num_sensors = 8
        self.sensor_angles = [0, 45, 90, 135, 180, 225, 270, 315]
        self.sensor_colors = [
            (255, 0, 0),  # Вперед (0°) - красный
            (200, 0, 100),  # Вперед-вправо (45°)
            (150, 0, 150),  # Вправо (90°)
            (100, 0, 200),  # Назад-вправо (135°)
            (0, 255, 0),  # Назад (180°) - зеленый
            (0, 100, 200),  # Назад-влево (225°)
            (0, 150, 150),  # Влево (270°)
            (0, 200, 100)  # Вперед-влево (315°)
        ]
        self.max_sensor_length = 300
        self.sensor_data = [0] * self.num_sensors

        self.clock = pygame.time.Clock() if self.render_mode == "human" else None

    def reset(self):
        """Сброс параметров к состоянию после инициализации."""
        self.car_x = self.track_data["start_position"]["x"]
        self.car_y = self.track_data["start_position"]["y"]
        self.car_angle = -math.degrees(math.atan(-(self.A / self.B)))
        self.velocity_angle = self.car_angle
        self.car_speed = 0
        self.laps = -1
        self.start_time = None
        self.is_invalid_lap = False
        self.previous_f_result = None
        self.best_lap_time = None
        self.last_valid_lap_time = None
        self.car_image = self.default_image

    @staticmethod
    def load_track_data(file_path, track_name):
        with open(file_path, "r") as f:
            data = json.load(f)
        for track in data["tracks"]:
            if track_name in track:
                return track[track_name]
        return None

    def cast_ray(self, angle):
        """Бросает луч под заданным углом и возвращает нормализованное расстояние"""
        start_x = self.car_x
        start_y = self.car_y
        rad = math.radians(self.car_angle + angle)

        max_len = self.max_sensor_length
        if 135 <= abs(angle) <= 225:
            max_len *= 0.7  # Укорачиваем задние лучи

        end_x = start_x + math.sin(rad) * max_len
        end_y = start_y + math.cos(rad) * max_len

        dx = end_x - start_x
        dy = end_y - start_y
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            return 0.0

        x_inc = dx / steps
        y_inc = dy / steps

        current_x = start_x
        current_y = start_y
        hit_distance = max_len  # По умолчанию максимальная длина

        for _ in range(int(steps)):
            ix = int(current_x)
            iy = int(current_y)

            # Проверка выхода за границы
            if not (0 <= ix < self.WIDTH and 0 <= iy < self.HEIGHT):
                edge_dist = self._distance_to_edge(start_x, start_y, current_x, current_y)
                hit_distance = min(hit_distance, edge_dist)
                break

            # Проверка пересечения с трассой
            if not self.on_track(ix, iy):
                dist = math.hypot(current_x - start_x, current_y - start_y)
                hit_distance = min(hit_distance, dist)
                break

            current_x += x_inc
            current_y += y_inc

        return hit_distance / max_len  # Нормализация

    def _distance_to_edge(self, x0, y0, x1, y1):
        """Вычисляет расстояние до ближайшей границы экрана"""
        t = 1.0
        dx = x1 - x0
        dy = y1 - y0

        if dx != 0:
            tx1 = (self.WIDTH - 1 - x0) / dx
            tx2 = -x0 / dx
            t_x = min(tx1, tx2) if dx > 0 else max(tx1, tx2)
            t = min(t, t_x)

        if dy != 0:
            ty1 = (self.HEIGHT - 1 - y0) / dy
            ty2 = -y0 / dy
            t_y = min(ty1, ty2) if dy > 0 else max(ty1, ty2)
            t = min(t, t_y)

        intersection_x = x0 + dx * t
        intersection_y = y0 + dy * t
        return math.hypot(intersection_x - x0, intersection_y - y0)

    def update_sensors(self):
        """Обновляет данные всех сенсоров"""
        self.sensor_data = [self.cast_ray(angle) for angle in self.sensor_angles]

    def update_physics(self, steering: int, throttle: int):
        """Обновление физики машины: steering=0:left,1:straight,2:right; throttle=0:reverse,1:idle,2:forward"""
        if self.render_mode == "human" and self.screen and self.track:
            self.screen.blit(self.track, (0, 0))

        # Apply steering
        if steering == 0:
            self.car_angle += self.rotation_speed
        elif steering == 2:
            self.car_angle -= self.rotation_speed
        # straight does nothing

        # Apply throttle
        if throttle == 2:
            self.car_speed = min(self.car_speed + self.acceleration, self.max_speed)
        elif throttle == 0:
            self.car_speed = max(
                self.car_speed - self.acceleration * 0.5, -self.max_speed
            )
        else:
            # idle / friction
            if abs(self.car_speed) < 0.1:
                self.car_speed = 0
            self.car_speed *= 0.995

        # Angle wrapping and drift
        if self.car_angle % 360 == 0:
            if self.car_angle > 0:
                self.velocity_angle = -(360 - self.velocity_angle)
            elif self.car_angle < 0:
                self.velocity_angle = 360 + self.velocity_angle
            self.car_angle = 0

        if self.velocity_angle < self.car_angle:
            self.velocity_angle += self.rotation_speed * (
                    1 - self.drift_factor * self.car_speed / self.max_speed
            )
            self.velocity_angle = max(self.velocity_angle, self.car_angle - 30)
        elif self.velocity_angle > self.car_angle:
            self.velocity_angle -= self.rotation_speed * (
                    1 - self.drift_factor * self.car_speed / self.max_speed
            )
            self.velocity_angle = min(self.velocity_angle, self.car_angle + 30)

        # Update position
        rad = math.radians(self.velocity_angle)
        self.car_x += -math.sin(rad) * self.car_speed
        self.car_y += -math.cos(rad) * self.car_speed

        # Boundaries
        self.car_x = max(min(self.car_x, self.WIDTH), 0)
        self.car_y = max(min(self.car_y, self.HEIGHT), 0)

        self.update_sensors()

        # Lap and track logic (unchanged)
        f_val = self.A * self.car_x + self.B * self.car_y + self.C
        f_result = -1 if f_val < 0 else 1

        if self.previous_f_result is None:
            self.previous_f_result = f_result

        if self.on_track(self.car_x, self.car_y):
            if not self.is_invalid_lap:
                self.car_image = self.default_image

            if self.previous_f_result < f_result and self.start_coords["x_min"] <= self.car_x <= self.start_coords["x_max"]:
                self.laps += 1
                if not self.is_invalid_lap and self.start_time is not None:
                    lap_time = time.perf_counter() - self.start_time

                    self.last_valid_lap_time = lap_time

                    if self.best_lap_time is None or lap_time < self.best_lap_time:
                        self.best_lap_time = lap_time

                self.start_time = time.perf_counter()
                self.is_invalid_lap = False
                self.car_image = self.default_image
        else:
            if not self.is_invalid_lap:
                self.car_image = self.dead_image

            self.is_invalid_lap = True

        self.previous_f_result = f_result

    def on_track(self, x, y):
        """Проверка, центр машины на трассе или на белой финишной линии."""
        x, y = int(x), int(y)

        # 1) Если центр за границами окна — точно не на трассе
        if x < 0 or x >= self.WIDTH or y < 0 or y >= self.HEIGHT:
            return False

        # 2) Финишная полоса (белый) считается частью трассы
        if (
                self.start_coords["x_min"] <= x <= self.start_coords["x_max"]
                and self.start_coords["y_min"] <= y <= self.start_coords["y_max"]
        ):
            return True

        if self.render_mode == "human" and self.screen is not None:
            pixel = self.screen.get_at((int(x), int(y)))[:3]
            return pixel == self.TRACK_COLOR or pixel[1] > 100

        if self.track_mask is not None:
            return bool(self.track_mask[int(y), int(x)])

    def scale_car_image(self, image, scale):
        if self.render_mode != "human" or image is None:
            return None

        new_w = int(self.car_width * scale)
        new_h = int(self.car_height * scale)
        return pygame.transform.scale(image, (new_w, new_h))

    @staticmethod
    def format_time(total_seconds):
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        ms = int((total_seconds - int(total_seconds)) * 1000)
        return f"{minutes}.{seconds:02}.{ms:03}"

    def render(self):
        """Отображение состояния игры."""
        if self.render_mode != "human":
            return

        pygame.event.pump()

        self.screen.blit(self.track, (0, 0))

        scaled = self.scale_car_image(self.car_image, self.car_scale)
        rotated = pygame.transform.rotate(scaled, self.car_angle)
        rect = rotated.get_rect(center=(self.car_x, self.car_y))
        self.screen.blit(rotated, rect.topleft)

        self.display_text()

        # Отрисовка сенсоров с разными цветами
        for i, (distance, color) in enumerate(zip(self.sensor_data, self.sensor_colors)):
            angle = self.car_angle + self.sensor_angles[i]
            rad = math.radians(angle)
            length = distance * self.max_sensor_length * (
                0.7 if 135 <= abs(self.sensor_angles[i]) <= 225 else 1.0
            )
            end_x = self.car_x + math.sin(rad) * length
            end_y = self.car_y + math.cos(rad) * length
            pygame.draw.line(self.screen, color,
                             (int(self.car_x), int(self.car_y)),
                             (int(end_x), int(end_y)), 2)

        # pygame.display.flip()

    def display_text(self):
        """Отображение данных круга и времени."""
        if self.render_mode != "human" or self.font is None:
            return

        laps_text = self.font.render(f"Круги: {self.laps}", True, self.WHITE)
        self.screen.blit(laps_text, (25, 50))

        if self.start_time is not None:
            cur_time = time.perf_counter() - self.start_time
            color = self.WHITE if not self.is_invalid_lap else self.RED
            cur_text = self.font.render(
                f"Текущий круг: {self.format_time(cur_time)}", True, color
            )
            self.screen.blit(cur_text, (125, 50))

        if self.best_lap_time is not None:
            best_text = self.font.render(
                f"Лучший круг: {self.format_time(self.best_lap_time)}", True, self.WHITE
            )
            self.screen.blit(best_text, (350, 50))

    def close(self):
        """Закрытие игры и освобождение ресурсов."""
        if self.render_mode == "human":
            pygame.quit()

        return
