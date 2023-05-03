#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict

import numpy as np

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_task import DynNavRLEnv


@registry.register_task(name="CatNavToObjTask-v0")
class CatDynNavRLEnv(DynNavRLEnv):
    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(
            config=config,
            *args,
            dataset=dataset,
            **kwargs,
        )

    def _generate_nav_to_pos(
        self, episode, start_hold_obj_idx=None, force_idx=None
    ):
        # learn nav to pick skill if not holding object currently
        if start_hold_obj_idx is None:
            # starting positions of candidate objects
            all_pos = np.stack(
                [
                    view_point.agent_state.position
                    for goal in episode.candidate_objects
                    for view_point in goal.view_points
                ],
                axis=0,
            )
            if force_idx is not None:
                raise NotImplementedError
        else:
            # positions of candidate goal receptacles
            all_pos = np.stack(
                [
                    view_point.agent_state.position
                    for goal in episode.candidate_goal_receps
                    for view_point in goal.view_points
                ],
                axis=0,
            )

        return all_pos