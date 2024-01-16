#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# must call this before importing habitat or magnum! avoids EGL_BAD_ACCESS error on some platforms
import ctypes
import sys

flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

import magnum as mn

from habitat_hitl._internal.config_helper import update_config
from habitat_hitl._internal.hitl_driver import HitlDriver
from habitat_hitl._internal.networking.average_rate_tracker import (
    AverageRateTracker,
)
from habitat_hitl._internal.networking.frequency_limiter import (
    FrequencyLimiter,
)
from habitat_hitl.core.mock_gui_input import (
    ConsoleTextDrawer,
    MockLineRender,
)
from habitat_hitl.core.gui_input import GuiInput


def _parse_debug_third_person(hitl_config, viewport_multiplier=(1, 1)):
    assert viewport_multiplier[0] > 0 and viewport_multiplier[1] > 0

    do_show = hitl_config.debug_third_person_viewport.width is not None

    if do_show:
        width = hitl_config.debug_third_person_viewport.width
        # default to square aspect ratio
        height = (
            hitl_config.debug_third_person_viewport.height
            if hitl_config.debug_third_person_viewport.height is not None
            else width
        )

        width = int(width * viewport_multiplier[0])
        height = int(height * viewport_multiplier[1])
    else:
        width = 0
        height = 0

    return do_show, width, height


def _headless_app_loop(hitl_config, driver):
    frequency_limiter = FrequencyLimiter(hitl_config.target_sps)
    rate_tracker = AverageRateTracker(1.0)
    dt = 1.0 / hitl_config.target_sps

    exit_after = None  # 500
    step_count = 0

    debug_video_images = None

    while True:
        post_sim_update_dict = driver.sim_update(dt)

        if "application_exit" in post_sim_update_dict:
            break

        # todo: save out debug images?
        if debug_video_images is None:
            debug_video_images = []
            for image in post_sim_update_dict["debug_images"]:
                debug_video_images.append([image])
        else:
            for image_idx, image in enumerate(
                post_sim_update_dict["debug_images"]
            ):
                assert debug_video_images[image_idx][0].shape == image.shape
                debug_video_images[image_idx].append(image)

        frequency_limiter.limit_frequency()

        new_rate = rate_tracker.increment()
        if new_rate is not None:
            print(f"SPS: {new_rate:.1f}")

        step_count += 1
        if exit_after is not None and step_count == exit_after:
            break

    if True:
        import os.path as osp

        import numpy as np

        from habitat_sim.utils import viz_utils as vut

        # all_obs = np.transpose(all_obs, (0, 2, 1, 3))  # type: ignore[assignment]
        save_video_dir = "./"
        # os.makedirs(SAVE_VIDEO_DIR, exist_ok=True)
        for image_idx, images in enumerate(debug_video_images):
            np_images = np.array(images)
            np_images = np_images[:, ::-1, :, :]  # flip vertically
            np_images = np.expand_dims(
                np_images, 1
            )  # add dummy dimension required by vut.make_video
            vut.make_video(
                np_images,
                0,
                "color",
                osp.join(save_video_dir, f"temp{image_idx}"),
                fps=10,
                open_vid=False,
            )

    driver.close()


def hitl_headless_main(config, create_app_state_lambda=None):
    hitl_config = config.habitat_hitl
    if "window" in hitl_config and hitl_config.window is not None:
        raise ValueError(
            "For habitat_hitl.headless=True, habitat_hitl.window should be None."
        )

    (
        show_debug_third_person,
        debug_third_person_width,
        debug_third_person_height,
    ) = _parse_debug_third_person(hitl_config)

    update_config(
        config,
        show_debug_third_person=show_debug_third_person,
        debug_third_person_width=debug_third_person_width,
        debug_third_person_height=debug_third_person_height,
    )

    driver = HitlDriver(
        config,
        GuiInput(),
        MockLineRender(),
        ConsoleTextDrawer(),
        create_app_state_lambda,
    )

    # sanity check if there are no agents with camera sensors
    if (
        len(config.habitat.simulator.agents) == 1
        and config.habitat_hitl.gui_controlled_agent.agent_index is not None
    ):
        assert driver.get_sim().renderer is None

    _headless_app_loop(hitl_config, driver)

    driver.close()


def hitl_main(config, create_app_state_lambda=None):
    hitl_config = config.habitat_hitl

    # todo: cleaner switch (headed_main?)
    if hitl_config.headless:
        hitl_headless_main(config, create_app_state_lambda)
        return

    from habitat_hitl._internal.gui_application import GuiApplication
    from magnum.platform.glfw import Application
    from habitat_hitl._internal.replay_gui_app_renderer import ReplayGuiAppRenderer

    glfw_config = Application.Configuration()
    glfw_config.title = hitl_config.window.title
    glfw_config.size = (hitl_config.window.width, hitl_config.window.height)
    gui_app_wrapper = GuiApplication(glfw_config, hitl_config.target_sps)
    # on Mac Retina displays, this will be 2x the window size
    framebuffer_size = gui_app_wrapper.get_framebuffer_size()

    viewport_multiplier = (
        framebuffer_size.x // hitl_config.window.width,
        framebuffer_size.y // hitl_config.window.height,
    )

    (
        show_debug_third_person,
        debug_third_person_width,
        debug_third_person_height,
    ) = _parse_debug_third_person(hitl_config, viewport_multiplier)

    viewport_rect = None
    if show_debug_third_person:
        # adjust main viewport to leave room for the debug third-person camera on the right
        assert framebuffer_size.x > debug_third_person_width
        viewport_rect = mn.Range2Di(
            mn.Vector2i(0, 0),
            mn.Vector2i(
                framebuffer_size.x - debug_third_person_width,
                framebuffer_size.y,
            ),
        )

    # note this must be created after GuiApplication due to OpenGL stuff
    app_renderer = ReplayGuiAppRenderer(
        framebuffer_size,
        viewport_rect,
        hitl_config.experimental.use_batch_renderer,
    )

    update_config(
        config,
        show_debug_third_person=show_debug_third_person,
        debug_third_person_width=debug_third_person_width,
        debug_third_person_height=debug_third_person_height,
    )

    driver = HitlDriver(
        config,
        gui_app_wrapper.get_sim_input(),
        app_renderer._replay_renderer.debug_line_render(0),
        app_renderer._text_drawer,
        create_app_state_lambda,
    )

    # sanity check if there are no agents with camera sensors
    if (
        len(config.habitat.simulator.agents) == 1
        and config.habitat_hitl.gui_controlled_agent.agent_index is not None
    ):
        assert driver.get_sim().renderer is None

    gui_app_wrapper.set_driver_and_renderer(driver, app_renderer)

    gui_app_wrapper.exec()

    driver.close()
