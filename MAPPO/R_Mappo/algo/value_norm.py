import torch
import torch.nn as nn


class ValueNorm(nn.Module):

    """
    Running normalization of critic targets, following the official
    MAPPO implementation (marlbenchmark/on-policy, valuenorm.py).

    The critic is trained against normalized returns, so its raw
    outputs live in normalized space; denormalize() maps them back to
    environment scale wherever they feed GAE / bootstrapping.
    """

    def __init__(
        self,
        input_shape=1,
        beta=0.99999,
        epsilon=1e-5
    ):

        super(ValueNorm, self).__init__()

        self.beta = beta
        self.epsilon = epsilon

        self.register_buffer(
            "running_mean",
            torch.zeros(input_shape)
        )

        self.register_buffer(
            "running_mean_sq",
            torch.zeros(input_shape)
        )

        self.register_buffer(
            "debiasing_term",
            torch.tensor(0.0)
        )

    def running_mean_var(self):

        debiased_mean = (
            self.running_mean
            / self.debiasing_term.clamp(min=self.epsilon)
        )

        debiased_mean_sq = (
            self.running_mean_sq
            / self.debiasing_term.clamp(min=self.epsilon)
        )

        debiased_var = (
            debiased_mean_sq - debiased_mean ** 2
        ).clamp(min=1e-2)

        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector):

        batch_mean = input_vector.mean()
        batch_sq_mean = (input_vector ** 2).mean()

        self.running_mean.mul_(self.beta).add_(
            batch_mean * (1.0 - self.beta)
        )

        self.running_mean_sq.mul_(self.beta).add_(
            batch_sq_mean * (1.0 - self.beta)
        )

        self.debiasing_term.mul_(self.beta).add_(
            1.0 * (1.0 - self.beta)
        )

    def normalize(self, input_vector):

        mean, var = self.running_mean_var()

        return (input_vector - mean) / torch.sqrt(var)

    def denormalize(self, input_vector):

        mean, var = self.running_mean_var()

        return input_vector * torch.sqrt(var) + mean
