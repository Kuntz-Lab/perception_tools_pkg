"""Frame-by-frame SAM 2 video tracking for live image streams.

SAM2VideoPredictor expects the whole video up front (init_state loads every frame). This drives the same
SAM2Base.track_step one frame at a time instead, and keeps only the memory SAM 2 reads: each object's prompted
(conditioning) frame plus the most recent frames, so memory use stays constant on an endless stream.

Several objects can be tracked at once. As in SAM2VideoPredictor, the image encoder runs once per frame and
each object is tracked with its own memory, so objects can be (re)prompted independently.

Based on SAM2CameraPredictor from https://github.com/Gy920/segment-anything-2-real-time, rebuilt on the
official sam2 package (box prompts).
"""

import numpy as np
import torch
import torch.nn.functional as F
from sam2.build_sam import build_sam2_video_predictor
from sam2.utils.misc import fill_holes_in_mask_scores

# SAM 2's ImageNet normalization (see sam2.utils.misc.load_video_frames)
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


class SAM2StreamTracker:
    def __init__(self, model_config, checkpoint_path, device="cuda"):
        self.device = torch.device(device)
        # The video predictor build adds the post-processing overrides used for video tracking; only the
        # SAM2Base methods it inherits are used here
        self.model = build_sam2_video_predictor(model_config, checkpoint_path, device=self.device)
        self.image_size = self.model.image_size
        self.img_mean = torch.tensor(IMG_MEAN, device=self.device).view(1, 3, 1, 1)
        self.img_std = torch.tensor(IMG_STD, device=self.device).view(1, 3, 1, 1)

        # Non-conditioning frames older than this are never read again: memory attention reads the last
        # (num_maskmem - 1) frames (strided), and the object pointers the last (max_obj_ptrs_in_encoder - 1)
        self.num_recent_frames = max(self.model.num_maskmem * self.model.memory_temporal_stride_for_eval,
                                     self.model.max_obj_ptrs_in_encoder)
        self.output_dicts = {}  # object id -> SAM 2 memory of that object
        self.frame_idx = -1

    @property
    def object_ids(self):
        """Ids of the objects being tracked."""
        return list(self.output_dicts)

    def remove(self, obj_id=None):
        """Stop tracking one object, or all objects if obj_id is None."""
        if obj_id is None:
            self.output_dicts.clear()
        else:
            self.output_dicts.pop(obj_id, None)

    @torch.inference_mode()
    def track(self, image, boxes=None):
        """Segment the tracked objects in the next RGB uint8 image.

        boxes maps object ids (any hashable) to boxes (x1, y1, x2, y2 in pixels) on this image. Each of those
        objects starts tracking from its box, replacing any earlier tracking of that id. Every other tracked
        object continues from its memory.

        Returns {obj_id: (mask, object_score)} for every tracked object: an HxW bool mask, and SAM 2's object
        score logit (> 0 when the object is visible; the mask is empty otherwise).
        """
        boxes = boxes or {}
        if not self.output_dicts and not boxes:
            raise RuntimeError("No objects to track; pass boxes to start tracking.")
        self.frame_idx += 1
        h, w = image.shape[:2]
        for obj_id in boxes:
            self.output_dicts[obj_id] = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}

        results = {}
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            # Image features are computed once and shared by all objects
            backbone_out = self.model.forward_image(self._preprocess(image))
            _, vision_feats, vision_pos_embeds, feat_sizes = self.model._prepare_backbone_features(backbone_out)

            for obj_id, output_dict in self.output_dicts.items():
                point_inputs = self._box_to_points(boxes[obj_id], w, h) if obj_id in boxes else None
                out = self.model.track_step(
                    frame_idx=self.frame_idx,
                    is_init_cond_frame=point_inputs is not None,
                    current_vision_feats=vision_feats,
                    current_vision_pos_embeds=vision_pos_embeds,
                    feat_sizes=feat_sizes,
                    point_inputs=point_inputs,
                    mask_inputs=None,
                    output_dict=output_dict,
                    # Offline this is the video length. A live stream has no end; any value at least
                    # max_obj_ptrs_in_encoder past the current frame behaves the same as a longer video
                    num_frames=self.frame_idx + self.model.max_obj_ptrs_in_encoder,
                    run_mem_encoder=True,
                )
                self._store_memory(output_dict, out, is_cond_frame=point_inputs is not None)
                results[obj_id] = (self._mask_at_image_size(out["pred_masks"], h, w),
                                   out["object_score_logits"].item())
        return results

    def _box_to_points(self, box, w, h):
        """A box prompt is its two corners as points with labels 2 and 3, in model input coordinates."""
        coords = torch.as_tensor(np.asarray(box, dtype=np.float32), device=self.device).view(1, 2, 2)
        coords = coords / torch.tensor([w, h], device=self.device) * self.image_size
        labels = torch.tensor([[2, 3]], dtype=torch.int32, device=self.device)
        return {"point_coords": coords, "point_labels": labels}

    def _store_memory(self, output_dict, out, is_cond_frame):
        """Keep only what later frames read from memory, as SAM2VideoPredictor does."""
        storage_key = "cond_frame_outputs" if is_cond_frame else "non_cond_frame_outputs"
        output_dict[storage_key][self.frame_idx] = {
            "maskmem_features": out["maskmem_features"].to(torch.bfloat16),
            "maskmem_pos_enc": out["maskmem_pos_enc"],
            "obj_ptr": out["obj_ptr"],
        }
        output_dict["non_cond_frame_outputs"].pop(self.frame_idx - self.num_recent_frames, None)

    def _mask_at_image_size(self, low_res_masks, h, w):
        low_res_masks = low_res_masks.float()
        if self.model.fill_hole_area > 0:
            low_res_masks = fill_holes_in_mask_scores(low_res_masks, self.model.fill_hole_area)
        mask = F.interpolate(low_res_masks, size=(h, w), mode="bilinear", align_corners=False)[0, 0] > 0
        return mask.cpu().numpy()

    def _preprocess(self, image):
        """RGB uint8 HxWx3 -> normalized 1x3xSxS float tensor at the model's input size."""
        img = torch.from_numpy(np.ascontiguousarray(image)).to(self.device).permute(2, 0, 1)[None].float() / 255.0
        img = F.interpolate(img, size=(self.image_size, self.image_size), mode="bilinear",
                            align_corners=False, antialias=True)
        return (img - self.img_mean) / self.img_std
