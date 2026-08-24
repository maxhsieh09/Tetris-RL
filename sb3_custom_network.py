import torch
from torch import nn
import torch.nn.functional as F

import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from tetris_env import TetrisEnv, make_vec_env, expand_actions


class BoardEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LazyConv2d(32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)), # Only reduce height
            nn.LazyConv2d(64, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, board):
        return self.encoder(board).mean(dim=3).flatten(start_dim=1)


class TetrisExtractor(BaseFeaturesExtractor):
    def __init__(self, obs_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(obs_space, features_dim=1)

        self.board_encoder = BoardEncoder()
        self.piece_encoder = nn.Linear(7, 32)

        self._features_dim = self.board_encoder(torch.zeros(1, 1, 10, 20)).numel()
        self._features_dim += self.piece_encoder(torch.zeros(1, 7)).numel()

    def forward(self, obs):
        board = obs["board"]
        piece = obs["piece"]

        if piece.ndim == 3:
            piece = piece.squeeze(1)

        board = self.board_encoder(board)
        piece = self.piece_encoder(piece)

        #print(board.shape, piece.shape)

        return torch.concat((board, piece), dim=-1)


env = TetrisEnv(render_mode="human", fps=1000)
env = expand_actions(env)
#env = make_vec_env(expanded_actions=True, render_mode="human", fps=1000)

#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

"""
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
"""

model = DQN(
    "MultiInputPolicy", env,
    learning_rate=3e-4,
    batch_size=256,
    target_update_interval=1000,
    policy_kwargs={"features_extractor_class": TetrisExtractor},
    verbose=1,
    #device=device
)

print(model.policy)
model.learn(total_timesteps=500000)

model.save("tetris_model")
