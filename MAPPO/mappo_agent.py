import torch
import torch.nn.functional as F
import numpy as np

from actor import Actor
from critic import Critic


class MAPPO:

    def __init__(
        self,
        obs_dim,
        global_state_dim,
        action_dim,
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.001,
        critic_coef=0.1,
        logger=None
    ):

        # ------------------------------------------------
        # Shared Actor
        # ------------------------------------------------

        self.actor = Actor(obs_dim, action_dim)

        # ------------------------------------------------
        # Centralized Critic
        # ------------------------------------------------

        self.critic = Critic(global_state_dim)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=lr_actor
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=lr_critic
        )

        # PPO Hyperparameters

        self.gamma = gamma

        self.gae_lambda = gae_lambda

        self.clip_eps = clip_eps

        self.entropy_coef = entropy_coef

        self.critic_coef = critic_coef
        self.logger = logger 
        

    def select_action(self, obs):

        obs = torch.FloatTensor(obs).unsqueeze(0)

        with torch.no_grad():

            action, log_prob = self.actor.get_action(obs)

        return (
            action.squeeze(0).numpy(),
            log_prob.item()
        )

    def compute_gae(
        self,
        rewards,
        values,
        dones,
        next_value
    ):

        advantages = []

        gae = 0

        values = list(values) + [next_value]

        for step in reversed(range(len(rewards))):

            delta = (
                rewards[step]
                +
                self.gamma
                * values[step + 1]
                * (1 - dones[step])
                -
                values[step]
            )

            gae = (
                delta
                +
                self.gamma
                * self.gae_lambda
                * (1 - dones[step])
                * gae
            )

            advantages.insert(0, gae)

        returns = [
            adv + val
            for adv, val in zip(advantages, values[:-1])
        ]

        return advantages, returns

    def update(
        self,
        obs_batch,
        global_state_batch,
        action_batch,
        old_logprob_batch,
        returns_batch,
        advantage_batch,
        epochs=10,
        batch_size=64
    ):

        obs_batch = torch.FloatTensor(obs_batch)

        global_state_batch = torch.FloatTensor(
            global_state_batch
        )

        action_batch = torch.FloatTensor(action_batch)

        old_logprob_batch = torch.FloatTensor(
            old_logprob_batch
        )

        returns_batch = torch.FloatTensor(
            returns_batch
        )

        advantage_batch = torch.FloatTensor(
            advantage_batch
        )

        # Normalize advantages
        advantage_batch = (
            advantage_batch - advantage_batch.mean()
        ) / (
            advantage_batch.std() + 1e-8
        )

        dataset_size = len(obs_batch)

        for _ in range(epochs):

            indices = np.random.permutation(dataset_size)

            for start in range(0, dataset_size, batch_size):

                end = start + batch_size

                batch_idx = indices[start:end]

                obs = obs_batch[batch_idx]

                global_states = global_state_batch[batch_idx]

                actions = action_batch[batch_idx]

                old_logprobs = old_logprob_batch[
                    batch_idx
                ]

                returns = returns_batch[batch_idx]

                advantages = advantage_batch[
                    batch_idx
                ]

                # -----------------------------------
                # PPO Actor Update
                # -----------------------------------

                new_logprobs, entropy = (
                    self.actor.evaluate(
                        obs,
                        actions
                    )
                )

                ratios = torch.exp(
                    new_logprobs - old_logprobs
                )
                approx_kl = (
                    old_logprobs - new_logprobs
                ).mean()
                if self.logger:
                    self.logger.info(
                        f"Ratio Mean: {ratios.mean().item():.4f} | "
                        f"KL: {approx_kl.item():.6f}"
                    )
                surr1 = ratios * advantages

                surr2 = torch.clamp(
                    ratios,
                    1 - self.clip_eps,
                    1 + self.clip_eps
                ) * advantages

                actor_loss = -torch.min(
                    surr1,
                    surr2
                ).mean()

                entropy_loss = -entropy.mean()

                # -----------------------------------
                # Critic Update
                # -----------------------------------

                values = self.critic(
                    global_states
                ).squeeze()

                critic_loss = F.mse_loss(
                    values,
                    returns
                )

                # -----------------------------------
                # Total Loss
                # -----------------------------------

                total_loss = (
                    actor_loss
                    +
                    self.critic_coef * critic_loss
                    +
                    self.entropy_coef * entropy_loss
                )

                self.actor_optimizer.zero_grad()

                self.critic_optimizer.zero_grad()

                total_loss.backward()

                # -----------------------------------
                # Gradient Clipping
                # -----------------------------------

                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    0.5
                )

                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    0.5
                )

                self.actor_optimizer.step()

                self.critic_optimizer.step()
                if self.logger:
                    self.logger.info(
                        f"Actor Loss: {actor_loss.item():.4f} | "
                        f"Critic Loss: {critic_loss.item():.4f} | "
                        f"Entropy: {entropy.mean().item():.4f}"
                    )