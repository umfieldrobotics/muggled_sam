#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This is a hack to make this script work from outside the root project folder (without requiring install)
try:
    import lib  # NOQA
except ModuleNotFoundError:
    import os
    import sys

    parent_folder = os.path.dirname(os.path.dirname(__file__))
    if "lib" in os.listdir(parent_folder):
        sys.path.insert(0, parent_folder)
    else:
        raise ImportError("Can't find path to lib folder!")

from collections import defaultdict
import cv2
import numpy as np
import torch
from lib.make_sam_v2 import make_samv2_from_original_state_dict
from lib.demo_helpers.video_data_storage import SAM2VideoObjectResults

# Imports for Mask2Former
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")
sys.path.append('../Mask2Former')
from mask2former import add_maskformer2_config

#####################
### Preliminaries ###
#####################

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    from sam2.build_sam import build_sam2_camera_predictor

global_idx = 0

def palette(i):
    a = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 165, 0),    # Orange
    (128, 0, 128),    # Purple
    (0, 255, 255),    # Cyan
    (255, 192, 203),  # Pink
    (128, 128, 128),  # Gray
    (255, 255, 255),  # White
    (128, 128, 0),    # Olive
    (255, 0, 255),    # Magenta
    (173, 216, 230),  # Light Blue
    (75, 0, 130),     # Indigo
    (165, 42, 42),    # Brown
    (0, 128, 128),    # Teal
    (255, 20, 147),   # Deep Pink
    (70, 130, 180)    # Steel Blue
    ]
    return a[i%18]

def run_seg_model(model, frame, masks, viz=False):
    global global_idx
    prompts_dict = {}
    
    # Run model
    outputs = model(frame)
    
    v = Visualizer(frame[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    v.draw_panoptic_seg(outputs["panoptic_seg"][0].to("cpu"), outputs["panoptic_seg"][1]).get_image()
    panoptics = outputs['panoptic_seg'][1]
    out_mask = outputs['panoptic_seg'][0].cpu().numpy()

    # Loop through predictions on frame
    for p in panoptics:
        if p['isthing'] == True: # Trackable things like car or motorcycle
            if p['area'] > 0.013*(frame.shape[0]*frame.shape[1]): # Discard out small things
                seg = np.where(out_mask == p['id'])
                seg_mask = np.where(out_mask == p['id'], 255, 0)
                
                # Generate two prompt points from the mask by finding the center of the mask and perturbing
                y_pom = abs(np.min(seg[0]) - np.max(seg[0])) // 6
                x_pom = abs(np.min(seg[1]) - np.max(seg[1])) // 6
                y = (np.min(seg[0]) + np.max(seg[0])) // 2
                x = (np.min(seg[1]) + np.max(seg[1])) // 2
                if masks is None:
                    prompts_dict[str(global_idx)] = {"fg_xy_norm_list": [[(x+x_pom)/frame.shape[1], (y-y_pom)/frame.shape[0]], [(x-x_pom)/frame.shape[1], (y+y_pom)/frame.shape[0]]], 
                                                     "bg_xy_norm_list": [], 
                                                     "box_tlbr_norm_list": []}
                    global_idx += 1
                elif (masks[y, x] == [0, 0, 0]).all():
                    prompts_dict[str(global_idx)] = {"fg_xy_norm_list": [[(x+x_pom)/frame.shape[1], (y-y_pom)/frame.shape[0]], [(x-x_pom)/frame.shape[1], (y+y_pom)/frame.shape[0]]], 
                                                     "bg_xy_norm_list": [], 
                                                     "box_tlbr_norm_list": []}
                    global_idx += 1
                if viz:
                    import matplotlib.pyplot as plt
                    inst_mask = np.where(masks == masks[y, x], 255, 0)
                    plt.imshow(frame.transpose(1,0,2), origin='lower')
                    plt.imshow(seg_mask.transpose(), origin='lower', alpha=0.5)
                    plt.imshow(inst_mask.transpose(1, 0, 2), origin='lower', alpha=0.5)
                    plt.scatter(y, x, s=20)
                    y1 = np.min(seg[0]); y2 = np.max(seg[0])
                    x1 = np.min(seg[1]); x2 = np.max(seg[1])
                    plt.scatter(y1, x1, s=20)
                    plt.scatter(y2, x2, s=20)
                    plt.show()
                    
    return prompts_dict


####################
### Run Settings ###
####################

# Define pathing & device usage
video_path = "./simple_examples/test.mp4"
model_path = "../sam2/checkpoints/sam2.1_hiera_large.pt"
device, dtype = "cpu", torch.float32
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16

# Define image processing config (shared for all video frames)
imgenc_config_dict = {"max_side_length": 1024, "use_square_sizing": True}

enable_prompt_visualization = False

# Read first frame to check that we can read from the video, then reset playback
vcap = cv2.VideoCapture(video_path)
ok_frame, first_frame = vcap.read()
if not ok_frame:
    raise IOError(f"Unable to read video frames: {video_path}")
vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)


#####################
### Set Up Models ###
#####################

# Set up SAM
print("Loading model...")
model_config_dict, sammodel = make_samv2_from_original_state_dict(model_path)
sammodel.to(device=device, dtype=dtype)

memory_per_obj_dict = defaultdict(SAM2VideoObjectResults.create)

# Set up segmentation model (Mask2Former)
cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file("../Mask2Former/configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl'
cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
model = DefaultPredictor(cfg)


############################
### Process Video Frames ###
############################

close_keycodes = {27, ord("q")}  # Esc or q to close
masks = None
bad_obj_score = {}
seg_freq = 1

try:
    total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_idx in range(total_frames):

        # Read frames
        ok_frame, frame = vcap.read()
        if not ok_frame:
            print("", "Done! No more frames...", sep="\n")
            break
        
        prompts_dict = {}
        if frame_idx % seg_freq == 0:
            prompts_dict = run_seg_model(model, frame, masks, viz=False)

        # Encode frame data (shared for all objects)
        encoded_imgs_list, _, _ = sammodel.encode_image(frame, **imgenc_config_dict)

        # Generate & store prompt memory encodings for each object as needed
        if prompts_dict is not None:

            # Loop over all sets of prompts for the current frame
            for obj_key_name, obj_prompts in prompts_dict.items():
                print(f"Generating prompt for object: {obj_key_name} (frame {frame_idx})")
                init_mask, init_mem, init_ptr = sammodel.initialize_video_masking(encoded_imgs_list, **obj_prompts)
                memory_per_obj_dict[obj_key_name].store_prompt_result(frame_idx, init_mem, init_ptr)

                # Draw prompts for debugging
                if enable_prompt_visualization:
                    prompt_vis_frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
                    norm_to_px_factor = np.float32((prompt_vis_frame.shape[1] - 1, prompt_vis_frame.shape[0] - 1))
                    for xy_norm in obj_prompts.get("fg_xy_norm_list", []):
                        xy_px = np.int32(xy_norm * norm_to_px_factor) # np.int32(xy_norm * norm_to_px_factor)
                        cv2.circle(prompt_vis_frame, xy_px, 3, (0, 255, 0), -1)
                    for xy_norm in obj_prompts.get("bg_xy_norm_list", []):
                        xy_px = np.int32(xy_norm * norm_to_px_factor)
                        cv2.circle(prompt_vis_frame, xy_px, 3, (0, 0, 255), -1)
                    for xy1_norm, xy2_norm in obj_prompts.get("box_tlbr_norm_list", []):
                        xy1_px = np.int32(xy1_norm * norm_to_px_factor)
                        xy2_px = np.int32(xy2_norm * norm_to_px_factor)
                        cv2.rectangle(prompt_vis_frame, xy1_px, xy2_px, (0, 255, 255), 2)

                    # Show prompt in it's own window and close after viewing
                    wintitle = f"Prompt ({obj_key_name}) - Press key to continue"
                    cv2.imshow(wintitle, prompt_vis_frame)
                    cv2.waitKey(0)
                    cv2.destroyWindow(wintitle)

        # Update tracking using newest frame
        combined_mask_result = np.zeros(frame.shape[0:2], dtype=bool)
        masks = np.zeros_like(frame) * 255
        for obj_key_name, obj_memory in memory_per_obj_dict.items():
            obj_score, best_mask_idx, mask_preds, mem_enc, obj_ptr = sammodel.step_video_masking(
                encoded_imgs_list, **obj_memory.to_dict()
            )

            # Skip storage for bad results (often due to occlusion)
            obj_score = obj_score.item()
            if obj_score < 0:
                # print(f"Bad object score for {obj_key_name}! Skipping memory storage...")
                if obj_key_name in bad_obj_score:
                    bad_obj_score[obj_key_name] += 1
                else:
                    bad_obj_score[obj_key_name] = 1
                continue

            # Store 'recent' memory encodings from current frame (helps track objects with changing appearance)
            # -> This can be commented out and tracking may still work, if object doesn't change much
            obj_memory.store_result(frame_idx, mem_enc, obj_ptr)

            # Add object mask prediction to 'combine' mask for display
            # -> This is just for visualization, not needed for tracking
            obj_mask = torch.nn.functional.interpolate(
                mask_preds[:, best_mask_idx, :, :],
                size=combined_mask_result.shape,
                mode="bilinear",
                align_corners=False,
            )
            obj_mask_binary = (obj_mask > 0.0).cpu().numpy().squeeze()
            # combined_mask_result = np.bitwise_or(combined_mask_result, obj_mask_binary)
            mask = np.repeat(obj_mask_binary[..., np.newaxis], 3, axis=2)
            mask = mask.astype('uint8')
            indices = np.argwhere(mask == [1, 1, 1])
            for i in indices:
                masks[i[0], i[1]] = palette(int(obj_key_name))
        
        # Remove objects that have bad scores after 3 frames
        for obj_key_name, freq in bad_obj_score.items():
            if freq >= 3:
                print("Removed {obj_key_name} from tracking list.")
                del memory_per_obj_dict[obj_key_name]
                bad_obj_score[obj_key_name] = 0

        # Combine original image & mask result side-by-side for display
        # combined_mask_result_uint8 = combined_mask_result.astype(np.uint8) * 255
        # disp_mask = cv2.cvtColor(combined_mask_result_uint8, cv2.COLOR_GRAY2BGR)
        disp_mask = masks
        for idx, (obj_key_name, obj_memory) in enumerate(memory_per_obj_dict.items()):
            cv2.putText(disp_mask, obj_key_name, (idx*40 + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, palette(int(obj_key_name)), 4)
        sidebyside_frame = np.hstack((frame, disp_mask))
        cv2.imwrite('./out/' + str(frame_idx) + '.png', sidebyside_frame)
        sidebyside_frame = cv2.resize(sidebyside_frame, dsize=None, fx=0.5, fy=0.5)

        # Show result
        cv2.imshow("Video Segmentation Result - q to quit", sidebyside_frame)
        keypress = cv2.waitKey(1) & 0xFF
        if keypress in close_keycodes:
            break

except Exception as err:
    raise err

except KeyboardInterrupt:
    print("Closed by ctrl+c!")

finally:
    vcap.release()
    cv2.destroyAllWindows()
