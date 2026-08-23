import random

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np

# Use (x, y) instead of (row, col) convention everywhere

_piece_coords = [
    [(0, 1), (1, 1), (2, 1), (3, 1)], # I
    [(0, 0), (0, 1), (1, 1), (2, 1)], # J
    [(0, 1), (1, 1), (2, 1), (2, 0)], # L
    [(0, 0), (1, 0), (1, 1), (0, 1)], # O
    [(0, 1), (1, 1), (1, 0), (2, 0)], # S
    [(0, 1), (1, 1), (2, 1), (1, 0)], # T
    [(0, 0), (1, 0), (1, 1), (2, 1)], # Z
]

_grid_sizes = [4, 3, 3, 2, 3, 3, 3]
PIECE_GRIDS = []
for coords, size in zip(_piece_coords, _grid_sizes):
    grid = np.zeros((size, size), dtype=np.uint8)
    for x, y in coords:
        grid[x, y] = 1
    PIECE_GRIDS.append(grid)

LINE_CLEAR_REWARDS = [0., 10., 30., 50., 80.]

BLOCK_SCALE = 20
BLOCK_BORDER = 1


class TetrisEnv(gym.Env[dict, np.ndarray]):
    def __init__(self, width=10, height=20, render_mode=None, fps=60, render_last_only=False):
        self.shape = (width, height)
        self.render_mode = render_mode
        self.fps = fps
        self.render_last_only = render_last_only

        self.observation_space = gym.spaces.Dict(
            {
                "board": gym.spaces.Box(low=0, high=1, shape=(1, *self.shape)),
                "piece": gym.spaces.Discrete(7)
            }
        )
        self.action_space = gym.spaces.MultiDiscrete([width, 4])  # x, rotation

        self.board = np.zeros(self.shape, dtype=np.uint8)
        self.piece_queue = []
        self.piece_idx = 0
        self.total_reward = 0.

        self.screen = None
        self.clock = None
        self.font = None

    def _get_obs(self):
        return {"board": self.board[None, :, :].astype(np.float32), "piece": self.piece_idx}

    def _get_info(self):
        return {"score": self.total_reward}

    def _sample_piece(self):
        if len(self.piece_queue) == 0:
            self.piece_queue = self.np_random.choice(7, 7, replace=False).tolist()
        return self.piece_queue.pop()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.board[:] = 0
        self.piece_queue = self.np_random.choice(7, 7, replace=False).tolist()
        self.piece_idx = self._sample_piece()
        self.total_reward = 0.

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), self._get_info()

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        x = action[0]
        rotation = action[1]

        piece = np.rot90(PIECE_GRIDS[self.piece_idx], rotation)
        piece = np.trim_zeros(piece)

        if self.shape[0] - x < piece.shape[0]:
            #overflow_columns = piece.shape[0] - (self.shape[0] - x)
            #overflow_blocks = piece[-overflow_columns:]

            # The piece exceeds the right boundary, invalid move
            #if overflow_blocks.sum() > 0:
            return self._get_obs(), -1., False, False, self._get_info()

            #piece = piece[:-overflow_columns] # Trim overflowing empty blocks

        # Drop the piece
        for y in range(self.shape[1] - piece.shape[1]):
            intersection = self.board[x:x+piece.shape[0], y:y+piece.shape[1]] * piece

            if intersection.sum() > 0:
                if y == 0: # Game over
                    if self.render_mode == "human" and self.render_last_only:
                        self.render()
                    return self._get_obs(), 0., True, False, self._get_info()
                else:
                    y -= 1
                    break
        
        self.board[x:x+piece.shape[0], y:y+piece.shape[1]] += piece

        reward = 0.5 # Drop reward

        # Clear lines
        full_mask = self.board.sum(axis=0) == self.shape[0]
        lines_cleared = full_mask.sum()
        new_board = self.board[:, ~full_mask]

        # Pad the top of the board to keep the same height
        self.board = np.concat([np.zeros((self.shape[0], lines_cleared), dtype=np.uint8), new_board], axis=1)

        reward += LINE_CLEAR_REWARDS[lines_cleared]

        self.piece_idx = self._sample_piece()
        self.total_reward += reward

        if self.render_mode == "human" and not self.render_last_only:
            self.render()

        return self._get_obs(), reward, False, False, self._get_info()

    def render(self):
        import pygame

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (self.shape[0] * BLOCK_SCALE, self.shape[1] * BLOCK_SCALE)
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.font is None:
            self.font = pygame.font.Font(pygame.font.get_default_font(), 16)

        self.screen.fill((10, 20, 30))

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                if self.board[x, y] > 0:
                    pygame.draw.rect(
                        self.screen,
                        (200, 200, 200),
                        pygame.Rect(
                            x * BLOCK_SCALE + BLOCK_BORDER,
                            y * BLOCK_SCALE + BLOCK_BORDER,
                            BLOCK_SCALE - BLOCK_BORDER * 2,
                            BLOCK_SCALE - BLOCK_BORDER * 2
                        ),
                    )

        text = self.font.render(f"Score: {self.total_reward :.2f}", True, (255, 255, 255))
        self.screen.blit(text, (10, 10))

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        if self.screen is not None:
            import pygame
        
            pygame.display.quit()
            pygame.quit()


def expand_actions(env):
    width = env.shape[0]
    return gym.wrappers.TransformAction(
        env, lambda action: np.array([action // 4, action % 4]),
        gym.spaces.Discrete(width * 4)
    )


check_env(TetrisEnv())

if __name__ == "__main__":
    env = TetrisEnv(render_mode="human", fps=10)
    env = expand_actions(env)
    env.reset()

    for _ in range(1000):
        state, reward, terminated, _, _ = env.step(env.action_space.sample())
        if terminated:
            env.reset()

    env.close()
