import json
import math
import sys
import time

import pygame

import config

pygame.init()

FPS = config.Config.FPS
WIDTH, HEIGHT = config.Config.WIDTH, config.Config.HEIGHT
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Машинка с дрифтом")

GREY = config.Config.GREY
TRACK_COLOR = config.Config.TRACK_COLOR
WHITE = config.Config.WHITE
RED = config.Config.RED

car_width, car_height = config.Config.CAR_WIDTH, config.Config.CAR_HEIGHT
car_speed = 0
max_speed = config.Config.MAX_SPEED
acceleration = config.Config.ACCELERATION
rotation_speed = config.Config.ROTATION_SPEED
drift_factor = config.Config.DRIFT_FACTOR

velocity_x = 0
velocity_y = 0

default_image = pygame.image.load("car.png").convert_alpha()
dead_image = pygame.image.load("red_car.png").convert_alpha()
car_image = default_image

font = pygame.font.SysFont(None, 24)


def load_track_data(file_path, track_name):
    with open(file_path, "r") as f:
        data = json.load(f)
    for track in data["tracks"]:
        if track_name in track:
            return track[track_name]
    return None


def draw_track(track_image):
    global screen
    screen.blit(track_image, (0, 0))


def on_track(x, y):
    global screen
    rv = False
    if 0 <= int(x) < WIDTH and 0 <= int(y) < HEIGHT:
        rv = screen.get_at((int(x), int(y)))[:3] == TRACK_COLOR
        rv |= screen.get_at((int(x), int(y)))[1] > 100
    return rv


def format_time(total_seconds):
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)
    return f"{minutes}.{seconds:02}.{milliseconds:03}"


def scale_car_image(car_image, car_scale):
    new_width = int(car_width * car_scale)
    new_height = int(car_height * car_scale)
    return pygame.transform.scale(car_image, (new_width, new_height))


if __name__ == "__main__":
    track_name = "track2"
    track_data = load_track_data("tracks.json", track_name)

    if track_data is None:
        print(f"Трек с именем '{track_name}' не найден в JSON файле.")
        sys.exit()

    track = pygame.image.load(track_data["track_image"]).convert()

    A = track_data["finish_line"]["A"]
    B = track_data["finish_line"]["B"]
    C = track_data["finish_line"]["C"]

    car_x, car_y = track_data["start_position"]["x"], track_data["start_position"]["y"]

    car_angle = -math.degrees(math.atan(-(A / B)))
    velocity_angle = car_angle

    laps = 0
    previous_f_result = None

    start_time = None
    last_valid_lap_time = None
    best_lap_time = None

    is_invalid_lap = False

    car_scale = config.Config.CAR_SCALE

    clock = pygame.time.Clock()
    running = True

    while running:
        clock.tick(FPS)
        draw_track(track)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            car_angle += rotation_speed
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            car_angle -= rotation_speed

        if keys[pygame.K_w] or keys[pygame.K_UP]:
            car_speed += acceleration
            if car_speed > max_speed:
                car_speed = max_speed
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            car_speed -= acceleration * 0.5
            if car_speed < -max_speed:
                car_speed = -max_speed
        else:
            if abs(car_speed) < 0.1:
                car_speed = 0
            car_speed *= 0.995

        if car_angle % 360 == 0:
            if car_angle > 0:
                velocity_angle = -(360 - velocity_angle)
            if car_angle < 0:
                velocity_angle = 360 + velocity_angle
            car_angle = 0

        if velocity_angle < car_angle:
            velocity_angle += rotation_speed * (
                1 - drift_factor * car_speed / max_speed
            )
            velocity_angle = max(velocity_angle, car_angle - 30)

        if velocity_angle > car_angle:
            velocity_angle -= rotation_speed * (
                1 - drift_factor * car_speed / max_speed
            )
            velocity_angle = min(velocity_angle, car_angle + 30)

        rad = math.radians(velocity_angle)
        velocity_x = -math.sin(rad) * car_speed
        velocity_y = -math.cos(rad) * car_speed

        car_x += velocity_x
        car_y += velocity_y

        car_x = max(min(car_x, WIDTH), 0)
        car_y = max(min(car_y, HEIGHT), 0)

        f_result = A * car_x + B * car_y + C
        f_result = -1 if f_result < 0 else 1

        if previous_f_result is None:
            previous_f_result = f_result

        if on_track(car_x, car_y):
            if not is_invalid_lap:
                car_image = default_image

            if previous_f_result < f_result and 1331 <= car_x <= 1374:
                laps += 1

                if not is_invalid_lap and start_time is not None:
                    end_time = time.perf_counter()
                    lap_time = end_time - start_time
                    last_valid_lap_time = lap_time
                    if best_lap_time is None or lap_time < best_lap_time:
                        best_lap_time = lap_time

                start_time = time.perf_counter()
                is_invalid_lap = False
                car_image = default_image
        else:
            if not is_invalid_lap:
                car_image = dead_image
            is_invalid_lap = True

        previous_f_result = f_result

        scaled_car_image = scale_car_image(car_image, car_scale)

        laps_text = font.render(f"Круги: {laps}", True, WHITE)
        screen.blit(laps_text, (25, 50))

        if start_time is not None:
            current_lap_time = time.perf_counter() - start_time
            lap_color = WHITE if not is_invalid_lap else RED
            current_lap_text = font.render(
                f"Текущий круг: {format_time(current_lap_time)}", True, lap_color
            )
            screen.blit(current_lap_text, (125, 50))

        if best_lap_time is not None:
            best_lap_text = font.render(
                f"Лучший круг: {format_time(best_lap_time)}", True, WHITE
            )
            screen.blit(best_lap_text, (350, 50))

        rotated_car = pygame.transform.rotate(scaled_car_image, car_angle)
        rotated_rect = rotated_car.get_rect(center=(car_x, car_y))
        screen.blit(rotated_car, rotated_rect.topleft)

        pygame.display.flip()

    pygame.quit()
    sys.exit()
