from functools import lru_cache


class ConfigMeta(type):
    """Метакласс, который позволяет динамически получать атрибуты."""

    def __getattr__(cls, name):
        # Проверка, существует ли атрибут
        if name not in cls.__dict__:
            raise AttributeError(f"Config не содержит атрибута '{name}'")
        return cls.__dict__[name]


class Config(metaclass=ConfigMeta):
    # Константы для игры
    FPS = 240
    WIDTH = 1536
    HEIGHT = 1024

    # Цвета
    GREY = (100, 100, 100)
    TRACK_COLOR = (180, 180, 180)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)

    # Параметры машинки
    CAR_WIDTH = 20
    CAR_HEIGHT = 40
    CAR_SCALE = 0.7

    # Скорость и управление
    ACCELERATION = 0.1 * 60 / FPS
    MAX_SPEED = 5 * 60 / FPS
    ROTATION_SPEED = 3 * 60 / FPS
    DRIFT_FACTOR = 0.6


@lru_cache()
def get_config() -> Config:
    return Config()
