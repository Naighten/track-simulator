import datetime
import os

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecCheckNan

from CarGameEnv import CarGameEnv
from callbacks import (
    ProgressCurriculumCallback,
    TargetedExploration,
    AdaptiveCurriculumCallback,
)


class TrainingConfig:
    # Конфигурация для GPU-оптимизированного обучения
    STAGES = {
        "initial": {
            "total_timesteps": 1_000_000,
            "n_envs": 12,  # Используем параллелизм GPU
            "n_steps": 8192,  # Крупные батчи для GPU
            "batch_size": 512,  # Оптимально для памяти GPU
            "learning_rate": 2.5e-4,
            "ent_coef": 0.3,
            "gamma": 0.99,
            "clip_range": 0.3,
            "network_size": [512, 512],
            "reward_tuning": {},
            "focus_areas": [],
            "network_migration": {},
        },
        "medium": {
            "total_timesteps": 3_000_000,
            "n_envs": 24,  # Максимальное использование GPU
            "n_steps": 4096,
            "batch_size": 1024,
            "learning_rate": 1e-4,
            "ent_coef": 0.1,
            "gamma": 0.995,
            "clip_range": 0.2,
            "network_size": [512, 512, 256],
            "reward_tuning": {},
            "focus_areas": [],
            "network_migration": {},
        },
        "final": {
            "total_timesteps": 4_000_000,
            "n_envs": 24,
            "n_steps": 2048,
            "batch_size": 2048,
            "learning_rate": 5e-5,
            "ent_coef": 0.01,
            "gamma": 0.998,
            "clip_range": 0.1,
            "network_size": [1024, 512, 256],
            "reward_tuning": {},
            "focus_areas": [],
            "network_migration": {},
        },
        "expert": {
            "total_timesteps": 5_000_000,
            "n_envs": 24,
            "n_steps": 2048,
            "batch_size": 2048,
            "learning_rate": 1e-5,  # Уменьшенный LR
            "ent_coef": 0.001,  # Меньше исследования
            "gamma": 0.999,
            "clip_range": 0.05,  # Более стабильная политика
            "network_size": [1024, 512, 256, 128],
            "reward_tuning": {  # Доп. настройки наград
                "checkpoint": 100.0,
                "proximity_coef": 0.8,
            },
            "focus_areas": [],
            "network_migration": {
                "expand_strategy": "depth",
                "weight_init": "orthogonal",
            },
        },
        "master": {
            "total_timesteps": 10_000_000,
            "n_envs": 32,
            "n_steps": 4096,
            "batch_size": 4096,
            "learning_rate": 5e-6,
            "ent_coef": 0.0001,
            "gamma": 0.9995,
            "clip_range": 0.02,
            "network_size": [2048, 1024, 512],
            "reward_tuning": {},
            "focus_areas": [
                (1195, 540),
                (890, 625),
                (760, 215),
                (625, 140),
                (380, 610),
                (395, 195),
                (95, 490),
                (415, 840),
                (850, 860),
                (1085, 940),
                (1360, 780),
                (1460, 755),
            ],  # Координаты проблемных зон
            "network_migration": {},
        },
    }


def train(stage_name, model=None, vec_normalize=None):
    config = TrainingConfig.STAGES[stage_name]

    # Добавим валидацию конфига перед использованием
    def validate_stage_config(config):
        if "network_size" in config:
            for val in config["network_size"]:
                if not isinstance(val, (int, float)):
                    raise TypeError(
                        f"Некорректный тип {type(val)} в network_size. Ожидается int/float"
                    )

    validate_stage_config(config)

    try:
        # Инициализация среды
        env = make_vec_env(
            lambda: CarGameEnv(
                track_name="track2",
                render_mode=None,
                reward_config=config["reward_tuning"] or {},
            ),
            n_envs=config["n_envs"],
        )
        env = VecCheckNan(env)

        # Блок обработки нормализации
        vecnorm_path = "./models/vecnormalize.pkl"
        if os.path.exists(vecnorm_path) and vec_normalize is None:
            env = VecNormalize.load(vecnorm_path, env)
            print("✅ Загружена нормализация из vecnormalize.pkl")
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=True, training=True)

        # Настройка системы наград
        if "reward_tuning" in config:
            env.env_method("update_rewards", config["reward_tuning"])

        # Блок загрузки модели
        model_path = "./models/latest.zip"
        if model is None and os.path.exists(model_path):
            print("✅ Загружаем модель")
            model = PPO.load(
                model_path,
                env=env,
                device="cuda",
                custom_objects={
                    "learning_rate": config["learning_rate"],
                    "clip_range": config["clip_range"],
                },
            )
            print(f"✅ Загружена модель из {model_path}")

            # Адаптация политики к новым размерам сети
            if "network_size" in config and model is not None:
                print(f"🔄 Адаптация архитектуры сети к {config['network_size']}")

                # Получаем текущие параметры политики
                policy_class = type(model.policy)
                old_policy_kwargs = model.policy_kwargs.copy()

                # Настройки миграции из конфига
                migration_cfg = config.get(
                    "network_migration",
                    {
                        "expand_strategy": "width",
                        "weight_init": "kaiming",
                        "freeze_existing": False,
                    },
                )

                # Обновляем параметры политики
                new_policy_kwargs = old_policy_kwargs.copy()
                new_policy_kwargs.update(
                    {
                        "net_arch": dict(
                            pi=config["network_size"], vf=config["network_size"]
                        )
                    }
                )

                # Создание новой модели
                new_model = PPO(
                    env=model.get_env(),
                    policy_kwargs=new_policy_kwargs,
                    device="cuda",
                    **model.get_parameters(),
                )

                # Перенос параметров
                old_params = model.policy.state_dict()
                new_params = new_model.policy.state_dict()

                # Инициализация новых параметров
                def init_weights(m, method):
                    if method == "kaiming":
                        torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    elif method == "orthogonal":
                        torch.nn.init.orthogonal_(m.weight)

                # Копирование весов
                for name, param in old_params.items():
                    if name in new_params:
                        new_params[name].copy_(param)
                        if migration_cfg["freeze_existing"]:
                            new_params[name].requires_grad = False

                # Инициализация новых слоев
                with torch.no_grad():
                    for name, param in new_model.policy.named_parameters():
                        if name not in old_params:
                            print(f"🧠 Инициализация новых параметров {name}")
                            init_weights(param, migration_cfg["weight_init"])

                new_model.policy.load_state_dict(new_params)

                print("✅ Архитектура успешно адаптирована")
                model = new_model

        if model is None:
            policy_kwargs = dict(
                net_arch=dict(pi=config["network_size"], vf=config["network_size"]),
                activation_fn=torch.nn.ReLU,
                ortho_init=True,
            )

            model = PPO(
                "MlpPolicy",
                env,
                verbose=2,
                device="cuda",
                n_steps=config["n_steps"],
                batch_size=config["batch_size"],
                learning_rate=config["learning_rate"],
                ent_coef=config["ent_coef"],
                gamma=config["gamma"],
                clip_range=config["clip_range"],
                vf_coef=0.5,
                max_grad_norm=0.7,
                policy_kwargs=policy_kwargs,
            )

        callbacks = [
            EvalCallback(
                eval_env=env,
                eval_freq=config["total_timesteps"] // 40,
                best_model_save_path="./models/best",
            ),
            ProgressCurriculumCallback(),
            TargetedExploration(config.get("focus_areas", [])),
            AdaptiveCurriculumCallback(config),
        ]

        # Начало обучения
        stage_start = datetime.datetime.now()
        print(f"\n⏳ Начало этапа '{stage_name}' в {stage_start:%Y-%m-%d %H:%M:%S}")

        try:
            model.learn(
                total_timesteps=config["total_timesteps"],
                callback=callbacks,
                reset_num_timesteps=False,
            )

        except Exception as e:
            # Сохранение при ошибке во время обучения
            error_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            error_model_path = f"./models/crash_{stage_name}_{error_time}.zip"
            error_vecnorm_path = f"./models/crash_vecnorm_{error_time}.pkl"

            print(f"\n⚠️ Критическая ошибка на этапе '{stage_name}': {str(e)}")
            print(f"💾 Экстренное сохранение модели в {error_model_path}")
            print(f"💾 Сохранение состояния нормализации в {error_vecnorm_path}")

            model.save(error_model_path)
            env.save(error_vecnorm_path)
            raise  # Пробрасываем исключение дальше

        finally:
            # Финализирующая часть всегда выполняется
            stage_end = datetime.datetime.now()
            duration = stage_end - stage_start
            print(f"\n✅ Этап '{stage_name}' завершен за {duration}")

            # Сохранение прогресса
            model.save("./models/latest")
            env.save("./models/vecnormalize.pkl")
            print(f"💾 Модель и нормализация сохранены в папке models")

    except Exception as e:
        # Глобальная обработка непредвиденных ошибок
        print(f"\n🔥 Необработанная ошибка: {str(e)}")
        print("🔄 Попробуйте уменьшить n_envs или batch_size")
        raise

    return model, env


if __name__ == "__main__":
    # Этапы обучения с прогрессивной настройкой
    model, vec_norm = train("initial")
    model, vec_norm = train("medium", model, vec_norm)
    model, vec_norm = train("final", model, vec_norm)
    model, vec_norm = train("expert", model, vec_norm)
    model, vec_norm = train("master", model, vec_norm)

    # Финальное сохранение
    model.save("./models/final_model")
    vec_norm.save("./models/final_vecnormalize.pkl")
