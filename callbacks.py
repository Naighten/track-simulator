import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class SaveOnSuccessCallback(BaseCallback):
    def __init__(self, save_path, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.success_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("success", False):
                self.success_count += 1
                model_path = os.path.join(
                    self.save_path, f"success_model_{self.success_count}"
                )
                self.model.save(model_path)
                if self.verbose:
                    print(
                        f"✅ Success #{self.success_count}: model saved to {model_path}"
                    )
        return True


class ProgressCurriculumCallback(BaseCallback):
    def __init__(self, progress_threshold: float = 0.7, verbose: int = 0):
        super().__init__(verbose)
        self.progress_threshold = progress_threshold
        self.best_progress = 0.0

    def _on_step(self) -> bool:
        env = self.training_env.envs[0].env
        current_progress = env.next_wp_idx / len(env.waypoints)

        if current_progress > self.best_progress:
            self.best_progress = current_progress
            if current_progress > self.progress_threshold:
                # Модифицируем награды для сложных участков
                env.reward_config.update(
                    {
                        "checkpoint": 75.0,
                        "proximity_coef": 0.7,
                        "angle_penalty_coef": 0.1,
                    }
                )
                print(
                    f"\n🔥 Активирован режим сложных участков (прогресс: {current_progress:.1%})"
                )
        return True


class TargetedExploration(BaseCallback):
    def __init__(self, focus_areas: list, verbose=0):
        super().__init__(verbose)
        self.focus_areas = focus_areas
        self.current_focus = 0

    def _on_step(self) -> bool:
        if len(self.focus_areas) == 0:
            return True
        # Для всех окружений в векторизированной среде
        for env_idx in range(self.training_env.num_envs):
            # Получаем доступ к объекту игры через цепочку атрибутов
            env = self.training_env.envs[env_idx]

            # Если используется VecNormalize или другие обертки
            while hasattr(env, "env"):
                env = env.env

            # Теперь напрямую обращаемся к атрибутам CarGame
            x = env.game.car_x
            y = env.game.car_y

            target = self.focus_areas[self.current_focus]
            if np.linalg.norm([x - target[0], y - target[1]]) < 50:
                # Обновляем параметры через метод окружения
                env.update_rewards({"offtrack_penalty": 3.0, "min_speed_penalty": 2.0})
                self.current_focus = (self.current_focus + 1) % len(self.focus_areas)
        return True


class AdaptiveCurriculumCallback(BaseCallback):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.stage_progress = 0

    def _on_step(self):
        # Динамическая регулировка параметров
        progress = self.num_timesteps / self.config["total_timesteps"]

        # Плавное уменьшение learning rate
        new_lr = self.config["learning_rate"] * (0.1**progress)
        self.model.learning_rate = new_lr

        # Адаптация энтропии
        if progress > 0.5:
            self.model.ent_coef = max(self.config["ent_coef"] * (1 - progress), 0.0001)

        return True
