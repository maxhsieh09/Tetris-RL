from stable_baselines3 import PPO

from tetris_env import TetrisEnv

env = TetrisEnv(render_mode="human", fps=1000, render_last_only=False)

model = PPO(
    "MultiInputPolicy", env,
    n_steps=256,
    batch_size=256,
    learning_rate=1e-3,
    #verbose=1
)
print(model.policy)
model.learn(total_timesteps=500000)

model.save("tetris_model")
