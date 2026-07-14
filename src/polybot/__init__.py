"""PolyBot: policy-value-belief self-play agent for PolyEnv."""

from .config import MCTSConfig, ModelConfig, TrainingConfig

__all__ = ["MCTSConfig", "ModelConfig", "PolicyValueBeliefNet", "TrainingConfig"]


def __getattr__(name: str):
    """Keep lightweight feature/log inspection usable before torch is installed."""
    if name == "PolicyValueBeliefNet":
        from .models.policy_value_belief import PolicyValueBeliefNet

        return PolicyValueBeliefNet
    raise AttributeError(name)
