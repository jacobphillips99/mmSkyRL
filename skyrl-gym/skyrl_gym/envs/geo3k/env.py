# we've actually overriden the MessageType, ConversationType in the generator/base
# should we copy those here? 
# likeweise, this isn't a text-in/text-out env!!!

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.geo3k import utils
from typing import Dict, Any
from omegaconf import DictConfig

class Geo3kEnv(BaseTextEnv):
    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()
        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"        
        self.ground_truth = extras["reward_spec"]["ground_truth"]

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True # one-step env
        reward = utils.compute_score(action, self.ground_truth)
        # nothing else to do
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={})
