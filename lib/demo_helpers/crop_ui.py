#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

import cv2
import numpy as np

from lib.demo_helpers.ui.window import DisplayWindow, KEY
from lib.demo_helpers.ui.layout import HStack, VStack, OverlayStack
from lib.demo_helpers.ui.sliders import HSlider
from lib.demo_helpers.ui.static import HSeparator, VSeparator, StaticMessageBar
from lib.demo_helpers.ui.images import ExpandingImage
from lib.demo_helpers.ui.buttons import ImmediateButton
from lib.demo_helpers.ui.overlays import CropBoxOverlay, HoverOverlay, DrawPolygonsOverlay
from lib.demo_helpers.ui.text import ValueBlock

from lib.demo_helpers.ui.helpers.images import get_image_hw_to_fill


# ---------------------------------------------------------------------------------------------------------------------
# %% Functions


def make_crop_ui(image_bgr, fg_line_color=(0, 255, 0)):
    """Function used to generate a simple UI for cropping images"""

    # Set up displays for showing image + zoomed area + cropped result
    main_disp = ExpandingImage(image_bgr)
    crop_disp = ExpandingImage(np.zeros((256, 256, 3), dtype=np.uint8))
    zoom_disp = ExpandingImage(np.zeros((256, 256, 3), dtype=np.uint8))
    zoom_poly_olay = DrawPolygonsOverlay(fg_line_color)

    # Set up interactive elements for user interactions
    zoom_olay = HoverOverlay()
    zoom_slider = HSlider("Zoom Factor", 0.5, 0, 1, step_size=0.05, marker_steps=5, enable_value_display=False)
    crop_olay = CropBoxOverlay(image_bgr.shape, fg_line_color, 2).set_box([(0.25, 0.25), (0.75, 0.75)])

    # Set up text blocks for feedback
    xy1_txt = ValueBlock("Crop XY1: ", "(0,0)")
    crop_wh_txt = ValueBlock("Crop WH: ", "(1,1)")
    xy2_txt = ValueBlock("Crop XY2: ", "(1,1)")
    done_btn = ImmediateButton("Done", color=(125, 185, 0))

    # Bundle all the ui elements
    img_h, img_w = image_bgr.shape[0:2]
    crop_ui = VStack(
        StaticMessageBar(
            f"Original: {img_w} x {img_h} px",
        ),
        HStack(
            OverlayStack(main_disp, zoom_olay, crop_olay),
            HSeparator(8, color=(40, 40, 40)),
            VStack(
                OverlayStack(zoom_disp, zoom_poly_olay),
                zoom_slider,
                VSeparator(8, color=(40, 40, 40)),
                crop_disp,
                done_btn,
            ),
        ),
        HStack(xy1_txt, crop_wh_txt, xy2_txt),
        StaticMessageBar(
            "Click & drag to adjust crop boundaries",
            "Arrow keys for fine adjustments",
            "Use ] or [ keys to zoom",
            text_scale=0.35,
            space_equally=True,
        ),
    )

    return (
        crop_ui,
        (zoom_olay, zoom_slider, crop_olay),
        (main_disp, zoom_disp, zoom_poly_olay, crop_disp, done_btn),
        (xy1_txt, crop_wh_txt, xy2_txt),
    )


# .....................................................................................................................


def run_crop_ui(
    image_bgr,
    render_height=800,
    fg_line_color=(0, 255, 0),
    bg_line_color=(0, 0, 0),
    window_title="Crop Image - q to close",
):
    """
    Helper used to launch a (temporary) UI for cropping an image
    Returns only the crop coords as slices:
        x_crop_slice, y_crop_slice

    To crop an image use:
        cropped_image = image[y_crop_slice, x_crop_slice, :]

    If the actual crop x/y values are needed, they can be accessed using:
        x1, x2 = x_crop_slice.start, x_crop_slice.stop
    """

    # Create & unpack ui elements
    crop_ui, ui_interact, ui_displays, ui_text = make_crop_ui(image_bgr, fg_line_color)
    zoom_olay, zoom_slider, crop_olay = ui_interact
    main_disp, zoom_disp, zoom_poly_olay, crop_disp, done_btn = ui_displays
    crop_xy1_txt, crop_wh_txt, crop_xy2_txt = ui_text

    # Set up window for display
    window = DisplayWindow(window_title)
    window.attach_mouse_callbacks(crop_ui)
    window.attach_keypress_callback("[", zoom_slider.decrement)
    window.attach_keypress_callback("]", zoom_slider.increment)
    window.attach_keypress_callback("w", lambda: crop_olay.nudge(up=1))
    window.attach_keypress_callback("s", lambda: crop_olay.nudge(down=1))
    window.attach_keypress_callback("a", lambda: crop_olay.nudge(left=1))
    window.attach_keypress_callback("d", lambda: crop_olay.nudge(right=1))
    window.attach_keypress_callback(KEY.UP_ARROW, lambda: crop_olay.nudge(up=1))
    window.attach_keypress_callback(KEY.DOWN_ARROW, lambda: crop_olay.nudge(down=1))
    window.attach_keypress_callback(KEY.LEFT_ARROW, lambda: crop_olay.nudge(left=1))
    window.attach_keypress_callback(KEY.RIGHT_ARROW, lambda: crop_olay.nudge(right=1))

    # Get scaling factors
    full_h, full_w = image_bgr.shape[0:2]
    max_wh_float = np.float32((full_w, full_h))
    zoom_boundary_px = 100
    min_zoom_xy = np.int32((zoom_boundary_px, zoom_boundary_px))
    max_zoom_xy = np.int32(max_wh_float) - min_zoom_xy

    # Initialize cropping to use full image by default
    crop_x1, crop_y1 = 0, 0
    crop_x2, crop_y2 = full_w, full_h

    try:
        while True:

            # Read overlays for user input
            is_zoom_changed, _, zoom_event_xy = zoom_olay.read()
            is_zoom_slider_changed, zoom_factor_norm = zoom_slider.read()
            is_crop_changed, is_valid_cropbox, crop_tlbr_norm = crop_olay.read()

            if is_zoom_slider_changed:
                zoom_boundary_px = 3 + int((1.0 - zoom_factor_norm) * 200)
                min_zoom_xy = np.int32((zoom_boundary_px, zoom_boundary_px))
                max_zoom_xy = np.int32(max_wh_float) - min_zoom_xy

            # If we don't have a valid box set, then just crop to the full image
            # -> Easier to reason about later, instead of always worrying about crop validity!
            if not is_valid_cropbox:
                crop_tlbr_norm = tuple(((0.0, 0.0), (1.0, 1.0)))

            # Update cropping coords whenever the crop changes
            if is_crop_changed:

                xy1_norm, xy2_norm = crop_tlbr_norm
                xy1_px = np.int32(np.round(xy1_norm * max_wh_float))
                xy2_px = np.int32(np.round(xy2_norm * max_wh_float))

                # Don't allow out-of-bounds cropping
                crop_x1, crop_y1 = np.clip(xy1_px, 0, np.int32(max_wh_float))
                crop_x2, crop_y2 = np.clip(xy2_px, 0, np.int32(max_wh_float))

                # Bail if crop is too small
                if (abs(crop_x2 - crop_x1) < 5) or (abs(crop_y2 - crop_y1) < 5):
                    crop_x1, crop_y1 = 0, 0
                    crop_x2, crop_y2 = np.int32(max_wh_float).tolist()

                # Crop the image!
                crop_image = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2, :]
                crop_w, crop_h = crop_image.shape[0:2]

                # Resize crop to fit into display area
                dispcrop_hw = crop_disp.get_render_hw()
                crop_scale_h, crop_scale_w = get_image_hw_to_fill(crop_image, dispcrop_hw)
                crop_scale_wh = (crop_scale_w, crop_scale_h)
                crop_image = cv2.resize(crop_image, dsize=crop_scale_wh, interpolation=cv2.INTER_NEAREST_EXACT)

                # Pad crop to match display area aspect ratio
                available_h, available_w = dispcrop_hw[0] - crop_scale_h, dispcrop_hw[1] - crop_scale_w
                pad_t, pad_l = available_h // 2, available_w // 2
                pad_b, pad_r = available_h - pad_t, available_w - pad_l
                crop_image = cv2.copyMakeBorder(crop_image, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT)
                crop_disp.set_image(crop_image)

                # Update text indicators
                crop_xy1_txt.set_value(f"({crop_x1}, {crop_y1})")
                crop_xy2_txt.set_value(f"({crop_x2}, {crop_y2})")
                crop_wh_txt.set_value(f"({crop_w}, {crop_h})")

            # Update zoom display when hover point changes
            if is_zoom_changed or is_zoom_slider_changed or is_crop_changed:

                # Calculate zoom region, taking into account frame boundaries
                zoom_xy_cen = np.int32(np.round(zoom_event_xy.xy_norm * max_wh_float))
                zoom_xy_cen = np.clip(zoom_xy_cen, min_zoom_xy, max_zoom_xy)
                zoom_x1, zoom_y1 = zoom_xy_cen - zoom_boundary_px
                zoom_x2, zoom_y2 = zoom_xy_cen + zoom_boundary_px + 1

                # Get zoomed (crop) of image around zoom point
                zoom_image = image_bgr[zoom_y1:zoom_y2, zoom_x1:zoom_x2, :].copy()

                # Draw crop line indicators into zoomed image
                zoom_h, zoom_w = zoom_image.shape[0:2]
                zoom_w_scale, zoom_h_scale = (zoom_w - 1, zoom_h - 1)
                zx1 = (crop_x1 - zoom_x1 - 1) / zoom_w_scale
                zy1 = (crop_y1 - zoom_y1 - 1) / zoom_h_scale
                zx2 = (crop_x2 - zoom_x1 + 1) / zoom_w_scale
                zy2 = (crop_y2 - zoom_y1 + 1) / zoom_h_scale
                zoom_poly_olay.set_polygons([[(zx1, zy1), (zx2, zy1), (zx2, zy2), (zx1, zy2)]])

                # Update displayed zoomed in image
                zoom_wh = tuple(reversed(zoom_disp.get_render_hw()))
                zoom_image = cv2.resize(zoom_image, dsize=zoom_wh, interpolation=cv2.INTER_NEAREST_EXACT)
                zoom_disp.set_image(zoom_image)

            # Update full display
            display_image = crop_ui.render(h=render_height)
            req_break, keypress = window.show(display_image)
            if req_break:
                break

            # Finish if user hits enter key (or q or esc)
            if keypress == KEY.ENTER:
                break

            # Finish when done is clicked
            is_done = done_btn.read()
            if is_done:
                break

    except KeyboardInterrupt:
        print("", "Crop cancelled with Ctrl+C", sep="\n")
        crop_x1, crop_y1 = 0, 0
        crop_x2, crop_y2 = full_w, full_h

    except Exception as err:
        raise err

    finally:
        window.close()

    # Bundle crop coords into convenient format for output
    x_crop_slice = slice(int(crop_x1), int(crop_x2))
    y_crop_slice = slice(int(crop_y1), int(crop_y2))

    return x_crop_slice, y_crop_slice