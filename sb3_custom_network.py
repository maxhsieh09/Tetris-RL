import torch.nn.functional as F

from stable_baselines3 import PPO, DQN

from policy_network import TetrisExtractor
from tetris_env import TetrisEnv, expand_actions


env = TetrisEnv(render_mode="human", fps=1000)
env = expand_actions(env)
#env = make_vec_env(expanded_actions=True, render_mode="human", fps=1000)

#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


model = PPO(
    "MultiInputPolicy", env,
    n_steps=256,
    batch_size=2048,
    learning_rate=3e-4,
    ent_coef=0.001,
    policy_kwargs={"features_extractor_class": TetrisExtractor},
    verbose=1,
    #device=device
)

'''
model = DQN(
    "MultiInputPolicy", env,
    learning_rate=3e-4,
    batch_size=256,
    target_update_interval=1000,
    policy_kwargs={"features_extractor_class": TetrisExtractor},
    verbose=1,
    #device=device
)
'''

print(model.policy)
model.learn(total_timesteps=500000)

model.save("tetris_model")
