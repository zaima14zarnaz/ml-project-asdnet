import numpy as np
from fpn_network import utils


def load_inference_data(dataset, image_id, config):
    pre_proc_data = dataset.load_obj_pre_proc_data(image_id)
    obj_feat = pre_proc_data["obj_feat"]
    p5_feat = pre_proc_data["P5"]

    object_roi_masks = dataset.load_object_roi_masks(image_id)

    # scale as before
    scale = 0.05
    padding = [(4, 4), (0, 0), (0, 0)]
    crop = None
    obj_spatial_masks = utils.resize_mask(object_roi_masks, scale, padding, crop)

    # guarantee a fixed 32x32:
    obj_spatial_masks_resized = np.zeros((32, 32, obj_spatial_masks.shape[2]), dtype=np.float32)
    h, w, n = obj_spatial_masks.shape
    # center-crop or pad to 32x32
    min_h = min(h, 32)
    min_w = min(w, 32)
    obj_spatial_masks_resized[:min_h, :min_w, :] = obj_spatial_masks[:min_h, :min_w, :]

    # transpose to [N, 32, 32, 1]
    obj_spatial_masks_resized = np.expand_dims(np.transpose(obj_spatial_masks_resized, [2, 0, 1]), -1)

    batch_obj_spatial_masks = np.zeros((config.SAL_OBJ_NUM, 32, 32, 1), dtype=np.float32)
    batch_obj_spatial_masks[:obj_spatial_masks_resized.shape[0]] = obj_spatial_masks_resized

    batch_obj_spatial_masks = np.expand_dims(batch_obj_spatial_masks, axis=0)

    return [obj_feat, batch_obj_spatial_masks, p5_feat]
