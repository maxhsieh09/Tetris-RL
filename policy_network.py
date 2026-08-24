import gymnasium as gym
import torch
from torch import nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FullPolicyNetwork(nn.Module):
    def __init__(self, use_two_heads=False, use_softmax=False):
        super().__init__()
        self.use_two_heads = use_two_heads
        self.use_softmax = use_softmax

        # Board is 10x20
        self.board_encoder = nn.Sequential(
            nn.LazyConv2d(32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)), # Only reduce height
            nn.LazyConv2d(64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.piece_encoder = nn.Embedding(7, 32)

        self.fc1 = nn.LazyLinear(128)
        self.policy_head = nn.LazyLinear(10 * 4)
        self.value_head = None
        if self.use_two_heads:
            self.value_head = nn.LazyLinear(1)

    def forward(self, obs):
        board = obs["board"]
        piece = obs["piece"]

        board = self.board_encoder(board).mean(dim=3).flatten(start_dim=1) # Reduce height dimension then flatten
        piece = self.piece_encoder(piece)

        x = torch.concat((board, piece), dim=-1)
        x = F.relu(self.fc1(x))

        policy = self.policy_head(x)
        if self.use_softmax:
            policy = F.softmax(policy, dim=-1)

        if self.use_two_heads:
            value = self.value_head(x)
            return policy, value
        else:
            return policy


if __name__ == "__main__":
    model = FullPolicyNetwork()
    sample_input = {
        "board": torch.randn((4, 1, 10, 20)),
        "piece": torch.randint(0, 7, (4,))
    }
    print(model(sample_input).shape)
    print(model)


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
