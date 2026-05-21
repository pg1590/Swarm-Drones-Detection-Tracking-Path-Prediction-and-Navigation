import torch
import torch.nn as nn
import torch.optim as optim

from algo.networks import RecurrentActor
from algo.networks import RecurrentCritic
from algo.networks import device


class R_MAPPO:

    def __init__(
        self,
        obs_dim,
        state_dim,
        action_dim,
        lr_actor=1e-4,
        lr_critic=1e-4,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        epochs=10,
        entropy_coef=0.01,
        value_coef=0.5,
        hidden_dim=128
    ):

        self.gamma = gamma
        self.lam = lam

        self.clip_eps = clip_eps

        self.epochs = epochs

        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        # =====================================================
        # NETWORKS
        # =====================================================

        self.actor = RecurrentActor(
            obs_dim,
            action_dim,
            hidden_dim
        ).to(device)

        self.critic = RecurrentCritic(
            state_dim,
            hidden_dim
        ).to(device)

        # =====================================================
        # OPTIMIZERS
        # =====================================================

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=lr_actor
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=lr_critic
        )

    # =========================================================
    # ACTION SELECTION
    # =========================================================

    def select_action(
        self,
        obs,
        state,
        actor_hidden,
        critic_hidden
    ):

        """
        obs:
            (obs_dim)

        Returns:
            action
            log_prob
            value
            next hidden states
        """

        obs_tensor = torch.FloatTensor(obs)\
            .unsqueeze(0)\
            .unsqueeze(0)\
            .to(device)

        # =====================================================
        # ACTOR
        # =====================================================

        dist, next_actor_hidden = self.actor(
            obs_tensor,
            actor_hidden
        )

        action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)

        # =====================================================
        # CRITIC
        # =====================================================

        state_tensor = torch.FloatTensor(state)\
            .unsqueeze(0)\
            .unsqueeze(0)\
            .to(device)

        value, next_critic_hidden = self.critic(
            state_tensor,
            critic_hidden
        )
        return (
            action.squeeze(0).squeeze(0).detach().cpu().numpy(),
            log_prob.item(),
            value.item(),
            next_actor_hidden.detach(),
            next_critic_hidden.detach()
        )

    # =========================================================
    # PPO UPDATE
    # =========================================================

    def update(self, buffer):

        (
            obs,
            states,
            actions,
            old_log_probs,
            values,
            returns,
            advantages,
            dones,
            actor_hidden_states,
            critic_hidden_states
        ) = buffer.get_tensors(device)

        # =====================================================
        # NORMALIZE ADVANTAGES
        # =====================================================

        advantages = (
            (advantages - advantages.mean())
            / (advantages.std() + 1e-8)
        )

        # =====================================================
        # RESHAPE FOR GRU
        # =====================================================

        obs = obs.unsqueeze(0)
        states = states.unsqueeze(0)
        actions = actions.unsqueeze(0)

        old_log_probs = old_log_probs.unsqueeze(0)

        returns = returns.unsqueeze(0)
        advantages = advantages.unsqueeze(0)

        # =====================================================
        # PPO EPOCHS
        # =====================================================

        for _ in range(self.epochs):

            # =================================================
            # ACTOR FORWARD
            # =================================================

            dist, _ = self.actor(
                obs,
                actor_hidden_states[:, 0:1, :]
            )

            new_log_probs = dist.log_prob(actions)\
                .sum(dim=-1)

            entropy = dist.entropy().sum(dim=-1).mean()

            # =================================================
            # PPO RATIO
            # =================================================

            ratio = torch.exp(
                new_log_probs - old_log_probs
            )

            surr1 = ratio * advantages

            surr2 = torch.clamp(
                ratio,
                1.0 - self.clip_eps,
                1.0 + self.clip_eps
            ) * advantages

            actor_loss = -torch.min(
                surr1,
                surr2
            ).mean()

            # =================================================
            # CRITIC FORWARD
            # =================================================

            values_pred, _ = self.critic(
                states,
                critic_hidden_states[:, 0:1, :]
            )

            values_pred = values_pred.squeeze(-1)

            critic_loss = (
                (returns - values_pred) ** 2
            ).mean()

            # =================================================
            # TOTAL LOSSES
            # =================================================

            total_actor_loss = (
                actor_loss
                - self.entropy_coef * entropy
            )

            total_critic_loss = (
                self.value_coef * critic_loss
            )

            # =================================================
            # ACTOR UPDATE
            # =================================================

            self.actor_optimizer.zero_grad()

            total_actor_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                0.5
            )

            self.actor_optimizer.step()

            # =================================================
            # CRITIC UPDATE
            # =================================================

            self.critic_optimizer.zero_grad()

            total_critic_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                0.5
            )

            self.critic_optimizer.step()

    # =========================================================
    # SAVE
    # =========================================================

    def save(self, path):

        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)

    # =========================================================
    # LOAD
    # =========================================================

    def load(self, path):

        checkpoint = torch.load(
            path,
            map_location=device
        )

        self.actor.load_state_dict(
            checkpoint['actor']
        )

        self.critic.load_state_dict(
            checkpoint['critic']
        )