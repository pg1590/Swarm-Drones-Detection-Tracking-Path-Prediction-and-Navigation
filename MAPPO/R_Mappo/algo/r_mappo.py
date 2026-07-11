import torch
import torch.nn as nn
import torch.optim as optim

from algo.networks import RecurrentActor
from algo.networks import RecurrentCritic
from algo.networks import device
from algo.value_norm import ValueNorm


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
        entropy_coef=0.03,
        value_coef=0.5,
        hidden_dim=128,
        sequence_length=16
    ):

        self.gamma = gamma
        self.lam = lam

        self.clip_eps = clip_eps

        self.epochs = epochs

        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.sequence_length = sequence_length
        self.action_epsilon = 1e-6

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

        # Canonical MAPPO ValueNorm: critic targets are normalized by
        # running statistics; critic outputs are denormalized wherever
        # they re-enter environment-scale computations (GAE).
        self.value_normalizer = ValueNorm(1).to(device)

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
        critic_hidden,
        deterministic=False
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

        if deterministic:
            raw_action = dist.mean
        else:
            raw_action = dist.rsample()

        action = torch.tanh(raw_action)

        log_prob = self._squashed_log_prob(
            dist,
            raw_action,
            action
        )

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

        value = self.value_normalizer.denormalize(value)

        return (
            action.squeeze(0).squeeze(0).detach().cpu().numpy(),
            log_prob.item(),
            value.item(),
            next_actor_hidden.detach(),
            next_critic_hidden.detach()
        )

    def get_value(self, state, critic_hidden):

        """
        Environment-scale value of a state, for GAE bootstrapping.
        """

        state_tensor = torch.FloatTensor(state)\
            .unsqueeze(0)\
            .unsqueeze(0)\
            .to(device)

        with torch.no_grad():

            value, _ = self.critic(
                state_tensor,
                critic_hidden
            )

            value = self.value_normalizer.denormalize(value)

        return value.item()

    def _atanh(self, action):

        action = torch.clamp(
            action,
            -1.0 + self.action_epsilon,
            1.0 - self.action_epsilon
        )

        return 0.5 * (
            torch.log1p(action)
            - torch.log1p(-action)
        )

    def _squashed_log_prob(
        self,
        dist,
        raw_action,
        action
    ):

        raw_log_prob = dist.log_prob(raw_action).sum(dim=-1)

        squash_correction = torch.log(
            1.0
            - action.pow(2)
            + self.action_epsilon
        ).sum(dim=-1)

        return raw_log_prob - squash_correction

    def _log_prob_from_squashed_action(
        self,
        dist,
        action
    ):

        action = torch.clamp(
            action,
            -1.0 + self.action_epsilon,
            1.0 - self.action_epsilon
        )

        raw_action = self._atanh(action)

        return self._squashed_log_prob(
            dist,
            raw_action,
            action
        )

    # =========================================================
    # PPO UPDATE
    # =========================================================

    def update(self, buffers):

        if not isinstance(buffers, (list, tuple)):
            buffers = [buffers]

        sequence_batches = []

        for buffer in buffers:
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

            sequence_batch = self._prepare_sequence_batch(
                obs,
                states,
                actions,
                old_log_probs,
                returns,
                advantages,
                dones,
                actor_hidden_states,
                critic_hidden_states
            )

            if sequence_batch is not None:
                sequence_batches.append(sequence_batch)

        if len(sequence_batches) == 0:
            return

        obs = torch.cat(
            [batch["obs"] for batch in sequence_batches],
            dim=0
        )

        states = torch.cat(
            [batch["states"] for batch in sequence_batches],
            dim=0
        )

        actions = torch.cat(
            [batch["actions"] for batch in sequence_batches],
            dim=0
        )

        old_log_probs = torch.cat(
            [batch["old_log_probs"] for batch in sequence_batches],
            dim=0
        )

        returns = torch.cat(
            [batch["returns"] for batch in sequence_batches],
            dim=0
        )

        advantages = torch.cat(
            [batch["advantages"] for batch in sequence_batches],
            dim=0
        )

        masks = torch.cat(
            [batch["masks"] for batch in sequence_batches],
            dim=0
        )

        initial_actor_hidden = torch.cat(
            [batch["actor_hidden"] for batch in sequence_batches],
            dim=1
        )

        initial_critic_hidden = torch.cat(
            [batch["critic_hidden"] for batch in sequence_batches],
            dim=1
        )

        # =====================================================
        # NORMALIZE ADVANTAGES
        # =====================================================

        advantages = (
            (advantages - advantages.mean())
            / (advantages.std() + 1e-8)
        )
        advantages = torch.clamp(
            advantages,
            -5.0,
            5.0
        )

        # =====================================================
        # PPO EPOCHS
        # =====================================================

        for _ in range(self.epochs):

            # =================================================
            # ACTOR FORWARD
            # =================================================

            dist, _ = self.actor(
                obs,
                initial_actor_hidden.contiguous(),
                masks
            )

            new_log_probs = self._log_prob_from_squashed_action(
                dist,
                actions
            )

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
                initial_critic_hidden.contiguous(),
                masks
            )

            values_pred = values_pred.squeeze(-1)

            # Critic regresses in normalized-return space; running
            # statistics are refreshed each epoch as in the official
            # trainer (cal_value_loss).
            self.value_normalizer.update(returns)

            normalized_returns = self.value_normalizer.normalize(returns)

            critic_loss = (
                (normalized_returns - values_pred) ** 2
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

    def _prepare_sequence_batch(
        self,
        obs,
        states,
        actions,
        old_log_probs,
        returns,
        advantages,
        dones,
        actor_hidden_states,
        critic_hidden_states
    ):

        seq_len = self.sequence_length
        num_steps = obs.shape[0]
        usable_steps = (num_steps // seq_len) * seq_len

        if usable_steps == 0:
            return None

        num_sequences = usable_steps // seq_len

        obs = obs[:usable_steps].view(
            num_sequences,
            seq_len,
            -1
        )

        states = states[:usable_steps].view(
            num_sequences,
            seq_len,
            -1
        )

        actions = actions[:usable_steps].view(
            num_sequences,
            seq_len,
            -1
        )

        old_log_probs = old_log_probs[:usable_steps].view(
            num_sequences,
            seq_len
        )

        returns = returns[:usable_steps].view(
            num_sequences,
            seq_len
        )

        advantages = advantages[:usable_steps].view(
            num_sequences,
            seq_len
        )

        # Hidden-state carry masks: step 0 of every chunk uses the
        # stored rollout hidden (already correct); later steps reset
        # the hidden wherever the previous step ended an episode.
        dones = dones[:usable_steps].view(
            num_sequences,
            seq_len
        )

        masks = torch.ones_like(dones)
        masks[:, 1:] = 1.0 - dones[:, :-1]

        actor_hidden = actor_hidden_states[
            :, :usable_steps, :
        ][:, ::seq_len, :]

        critic_hidden = critic_hidden_states[
            :, :usable_steps, :
        ][:, ::seq_len, :]

        return {
            "obs": obs,
            "states": states,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "returns": returns,
            "advantages": advantages,
            "masks": masks,
            "actor_hidden": actor_hidden,
            "critic_hidden": critic_hidden
        }

    # =========================================================
    # SAVE
    # =========================================================

    def save(self, path):

        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'value_norm': self.value_normalizer.state_dict()
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

        # Older checkpoints (pre-ValueNorm) have no stats; the critic
        # they contain was trained on raw returns, so fresh statistics
        # are the least-wrong option for continued use.
        if 'value_norm' in checkpoint:
            self.value_normalizer.load_state_dict(
                checkpoint['value_norm']
            )
