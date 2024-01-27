import os
import random
import sys

import git
import numpy as np
from gym import spaces

# %matplotlib inline
from matplotlib import pyplot as plt

from PIL import Image

import habitat
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.nav import NavigationTask
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config as get_baselines_config
from habitat_sim.utils.common import d3_40_colors_rgb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from gym import spaces
# import habitat_baselines.rl.ddppo.policy as pol
from habitat_baselines.rl.ddppo.policy import ( 
    PointNavResNetNet,
    PointNavResNetPolicy,
)
from habitat_baselines.config.default import get_config
from typing import Dict, Optional
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

if __name__ == "__main__":
    config = habitat.get_config(
        config_path="benchmark/rearrange/pick.yaml",
        overrides=[

            "habitat.environment.iterator_options.shuffle=False",
        ],
    )

    try:
        env.close()  # type: ignore[has-type]
    except NameError:
        pass
    env = habitat.Env(config=config)
from habitat.core.dataset import Dataset, Episode

dataset = env._dataset
def filter_fn(episode: Episode) -> bool:
    return int(episode.episode_id) < 3 


filtered_dataset = dataset.filter_episodes(filter_fn)
assert len(filtered_dataset.episodes) == 3
for ep in filtered_dataset.episodes:
        print(ep.info)
        assert filter_fn(ep)
env._dataset = filtered_dataset

def display_sample(
    rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):  # noqa: B006
    

    rgb_img = Image.fromarray(rgb_obs, mode="RGB")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new(
            "P", (semantic_obs.shape[1], semantic_obs.shape[0])
        )
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray(
            (depth_obs / 10 * 255).astype(np.uint8), mode="L"
        )
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.show(block=False)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from gym import spaces
# import habitat_baselines.rl.ddppo.policy as pol
from habitat_baselines.rl.ddppo.policy import ( 
    PointNavResNetNet,
    PointNavResNetPolicy,
)
from habitat_baselines.config.default import get_config
from typing import Dict, Optional
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
class MyPolicy(nn.Module):
    def __init__(self, **kwargs):
        super(MyPolicy, self).__init__()

        ACTION_SPACE = spaces.Discrete(4)

        OBSERVATION_SPACES = {
            "depth_model": spaces.Dict(
                {
                    "depth": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(224, 224, 1),
                        dtype=np.float32,
                    ),
                    "pointgoal_with_gps_compass": spaces.Box(
                        low=-3.4028235e+38,
                        high=3.4028235e+38,
                        shape=(2,),
                        dtype=np.float32,
                    ),
                }
            )
        }
        # config = get_config(
        #     "test/config/habitat_baselines/ddppo_pointnav_test.yaml"
        # )
        MODELS = {
            "pointnav_weights.pth": {
                "backbone": "resnet18",
                "observation_space": OBSERVATION_SPACES["depth_model"],
                "action_space": ACTION_SPACE,
            }}
        PTH_GPU_ID: int = 0
        self.device = (
            torch.device("cuda:{}".format(PTH_GPU_ID))
            # if torch.cuda.is_available()
            # else torch.device("cpu")
        )
        model_weights_path = '/nethome/asingh3064/flash/habitat-lab/examples/tutorials/pointnav_weights.pth'  # Replace with the actual path to your model weights file

        pretrained_state = torch.load(model_weights_path, map_location="cuda:0")

        # self.policy = PointNavResNetPolicy.from_config( config=config,
        #                                                 observation_space=OBSERVATION_SPACES["depth_model"], 
        #                                                 action_space=ACTION_SPACE)
        
        self.policy= PointNavResNetPolicy(observation_space=OBSERVATION_SPACES["depth_model"], 
                                                        action_space=ACTION_SPACE,
                                                        hidden_size = 512,
                                                        num_recurrent_layers = 2,
                                                        rnn_type="LSTM")
                                        
        self.policy.load_state_dict(pretrained_state)
        self.rnn_hidden_states: Optional[torch.Tensor] = None
        self.not_done_masks: Optional[torch.Tensor] = None
        self.prev_actions: Optional[torch.Tensor] = None
        # self.policy.to(self.device)
        self.hidden_size = 512
        self.policy.eval()
        self.prev_actions = torch.zeros(
            1, 1, dtype=torch.long, device='cuda:0')

        self.rnn_hidden_states = torch.zeros(
            1,
            self.policy.net.num_recurrent_layers,
            self.hidden_size,
            device=self.device,
        )
        self.not_done_masks = torch.zeros(
            1, 1, device='cuda:0', dtype=torch.bool
        )


        
    def forward(self, obs):
        # Extract observations
        depth_image = obs['head_depth']
        import pdb
        pdb.set_trace()
        print(env.sim.get_agent_state())
        print(env.sim.target_start_pos)

        relative_position = obs["obj_start_sensor"] 
        print(relative_position)
        rho = np.linalg.norm(relative_position)
        theta = np.arctan2(relative_position[1], relative_position[0])
        display_sample(obs["head_depth"])

        depth_image_tensor = torch.from_numpy(depth_image.transpose(2, 0, 1)).unsqueeze(0).float()  # Transpose to (1, 256, 256)tmux ls
        depth_image_tensor = depth_image_tensor.permute(0, 1, 3, 2)  # Transpose to (1, 256, 256)
        # print(depth_image_tensor.size())

        # Define the transformation to resize the image
        with torch.no_grad():
            resized_depth_image_tensor = F.interpolate(depth_image_tensor, size=(224, 224), mode='bilinear', align_corners=False)
            # resized_depth_image_tensor /= 5.0       
        resized_depth_image_tensor = resized_depth_image_tensor.permute(0,2,3,1)  
        # print(resized_depth_image_tensor.size())
        device = torch.device("cuda:0")  # Assuming you want to use GPU 0
        resized_depth_image_tensor = resized_depth_image_tensor.to(self.device)

        self.policy.to(device = 'cuda:0')
  
        print(resized_depth_image_tensor.device)
        print("Destination, distance: {:.3f}, theta (radians): {:.2f}".format(
            rho,
            theta))

        resized_point_goal_tensor = torch.tensor([[rho, theta]], dtype=torch.float32, device='cuda:0')        # Process observations through the policy
        action_logits = self.policy.act(
            observations={"depth": resized_depth_image_tensor, "pointgoal_with_gps_compass": resized_point_goal_tensor},
            rnn_hidden_states=self.rnn_hidden_states,
            prev_actions=self.prev_actions,
            masks=self.not_done_masks,
            deterministic=False)
        self.rnn_hidden_states = action_logits.rnn_hidden_states
        # Convert action logits to actions (0: stop, 1: move forward, 2: pivot left, 3: pivot right)
        action = torch.argmax(action_logits.actions, dim=1).item()
        values = torch.argmax(action_logits.values, dim=1).item()
        print("Action: ",action)
        display_sample(obs["head_rgb"])

        return action
    


pol=MyPolicy()
obs=env.reset()
action = pol.forward(obs)
action = 1

valid_actions = ["stop","move_forward","turn_left", "turn_right"]
action = valid_actions[action]
# print(env.observation_space)
interactive_control = False  # @param {type:"boolean"}
while not env.episode_over:
    observations = env.step(env.action_space.sample())  # noqa: F841
    info = env.get_metrics()
    print(env.current_episode.episode_id)
    while action != "stop":
        display_sample(obs["head_rgb"])
        obs = env.step(
            {
                "action": action,
                "action_args": None,
            }
        )
        action = pol.forward(obs)
        action = valid_actions[action]
        print(action)


env.close()