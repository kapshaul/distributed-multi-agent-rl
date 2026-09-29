# env_registry.py

from typing import Set

# Registry of environments typically using image-based observations
IMAGE_OBS_ENVS: Set[str] = {
    "ALE/Pong-v5",
    "ALE/SpaceInvaders-v5",
    "ALE/Adventure-v5",
    "ALE/Alien-v5",
    "ALE/Asteroids-v5",
    "ALE/AirRaid-v5",
    "ALE/BeamRider-v5",
    "ALE/Breakout-v5",
    "ALE/Enduro-v5",
    "ALE/Freeway-v5",
    "ALE/Frostbite-v5",
    "ALE/Gopher-v5",
    "ALE/Gravitar-v5",
    "ALE/Jamesbond-v5",
    "ALE/Kangaroo-v5",
    "ALE/Krull-v5",
    "CarRacing-v2",
    "MiniGrid-Empty-5x5-v0",
}

def is_image_observation_env(env_id: str) -> bool:
    """
    Checks whether the environment uses image observations,
    which generally require a CNN-based model.

    Args:
        env_id (str): Environment ID (e.g., 'Pong-v5').

    Returns:
        bool: True if the environment uses image observations.
    """
    return env_id in IMAGE_OBS_ENVS
