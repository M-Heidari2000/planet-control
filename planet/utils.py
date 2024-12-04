import numpy as np
from typing import Tuple


class ReplayBuffer:
    """
        Replay buffer holds sample trajectories
    """
    def __init__(
        self,
        capacity: int,
        observation_shape: int,
        action_shape: int,
    ):
        self.capacity = capacity

        self.observations = np.zeros((capacity, observation_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_shape), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=bool)

        self.index = 0
        self.is_filled = False

    def __len__(self):
        return self.capacity if self.is_filled else self.index

    def push(
        self,
        observation,
        action,
        reward,
        done,
    ):
        """
            Add experience (single step) to the replay buffer
        """
        self.observations[self.index] = observation
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.done[self.index] = done

        self.index = (self.index + 1) % self.capacity
        self.is_filled = self.is_filled or self.index == 0

    def sample(
        self,
        batch_size: int,
        chunk_length: int
    ):
        done = self.done.copy()
        done[-1] = 1
        episode_ends = np.where(done)[0]

        all_indexes = np.arange(len(self))
        distances = episode_ends[np.searchsorted(episode_ends, all_indexes)] - all_indexes + 1
        valid_indexes = all_indexes[distances >= chunk_length]

        sampled_indexes = np.random.choice(valid_indexes, size=batch_size)
        sampled_ranges = np.vstack([
            np.arange(start, start + chunk_length) for start in sampled_indexes
        ])

        sampled_observations = self.observations[sampled_ranges].reshape(
            batch_size, chunk_length, self.observations.shape[1]
        )
        sampled_actions = self.actions[sampled_ranges].reshape(
            batch_size, chunk_length, self.actions.shape[1]
        )
        sampled_rewards = self.rewards[sampled_ranges].reshape(
            batch_size, chunk_length, 1
        )
        sampled_done = self.done[sampled_ranges].reshape(
            batch_size, chunk_length, 1
        )

        return sampled_observations, sampled_actions, sampled_rewards, sampled_done


class ReplayBufferOriginal:

    """
        Replay Buffer holds sample trajectories
    """
    
    def __init__(
        self,
        capacity: int,
        observation_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
    ):
        self.capacity = capacity

        self.observations = np.zeros((capacity, observation_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_shape), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=bool)

        self.index = 0
        self.is_filled = False

    def __len__(self):
        return self.capacity if self.is_filled else self.index

    def push(
        self,
        observation,
        action,
        reward,
        done,
    ):
        """
            Add experience (single step) to the replay buffer
        """
        self.observations[self.index] = observation
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.done[self.index] = done

        if self.index == self.capacity - 1:
            self.is_filled = True
        self.index = (self.index + 1) % self.capacity

    def sample(
        self,
        batch_size: int,
        chunk_length: int
    ):
        """
            Sample trajectories from replay buffer
            batch_size and chunk_length are represented as B and L in the paper
            Each sample is a consecutive sequence
        """
        # Find where the episodes ended
        episode_borders = np.where(self.done)[0]
        sampled_indexes = []
        for _ in range(batch_size):
            # We are actially trying to sample a sequence of length L from just one episode
            # (No episode borders must be in between the inital_index and final_index)
            cross_border = True
            while cross_border:
                initial_index = np.random.randint(len(self) - chunk_length + 1)
                final_index = initial_index + chunk_length - 1
                cross_border = np.logical_and(
                    episode_borders >= initial_index,
                    episode_borders < final_index
                ).any()
            sampled_indexes += list(range(initial_index, final_index+1))

        sampled_observations = self.observations[sampled_indexes].reshape(
            batch_size, chunk_length, *self.observations.shape[1:]
        )
        sampled_actions = self.actions[sampled_indexes].reshape(
            batch_size, chunk_length, self.actions.shape[1]
        )
        sampled_rewards = self.rewards[sampled_indexes].reshape(
            batch_size, chunk_length, 1
        )
        sampled_done = self.done[sampled_indexes].reshape(
            batch_size, chunk_length, 1
        )

        return sampled_observations, sampled_actions, sampled_rewards, sampled_done