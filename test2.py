from stable_baselines3 import PPO

from CarGameEnv import CarGameEnv

MODEL_PATH = "ppo_cargame.zip"

agent = PPO.load(MODEL_PATH, device="cpu")
env = CarGameEnv(track_name="track2", render_mode="human")

NUM_EPISODES = 50
MAX_STEPS = 1000

for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
    total_reward = 0
    for step in range(MAX_STEPS):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        env.game.clock.tick(env.game.FPS)
        if done or truncated:
            print(
                f"Ep {ep + 1} ended at {step + 1}, reward={total_reward:.2f}, success={info.get('success')}"
            )
            break
env.close()
