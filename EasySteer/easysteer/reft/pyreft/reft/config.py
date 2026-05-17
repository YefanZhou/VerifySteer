from ..core import IntervenableConfig
import json


class ReftConfig(IntervenableConfig):
    """
    Reft config for Reft methods.
    """
    def __init__(
        self, **kwargs,
    ):
        super().__init__(**kwargs)