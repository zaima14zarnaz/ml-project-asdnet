import numpy as np
from fpn_network import utils
import logging


def data_generator(dataset, config, shuffle=True, augmentation=None, batch_size=1):
    b = 0
    image_index = -1
    image_ids = np.copy(dataset.img_ids)
    error_count = 0
    consecutive_errors = 0

    # Keras requires a generator to run indefinitely.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            image_id = image_ids[image_index]

            # Get data
            obj_feat, obj_spatial_masks, p5_feat, gt_ranks = load_image_gt(dataset, config, image_id,
                                                                           augmentation=augmentation)

            if not np.any(gt_ranks > 0):
                continue

            # Init batch arrays
            if b == 0:
                batch_obj_feat = np.zeros(
                    (batch_size, config.SAL_OBJ_NUM, 1, 1, config.OBJ_FEAT_SIZE), dtype=np.int32)

                batch_obj_spatial_masks = np.zeros(
                    (batch_size, config.SAL_OBJ_NUM, 32, 32, 1), dtype=np.float32)

                batch_p5_feat = np.zeros(
                    (batch_size, 32, 32, 256), dtype=np.float32)

                batch_gt_ranks = np.zeros((batch_size,) + gt_ranks.shape, dtype=gt_ranks.dtype)

            # Add to batch
            batch_obj_feat[b] = obj_feat
            batch_obj_spatial_masks[b] = obj_spatial_masks
            batch_p5_feat[b] = p5_feat

            batch_gt_ranks[b, :gt_ranks.shape[0]] = gt_ranks

            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = [batch_obj_feat, batch_obj_spatial_masks, batch_p5_feat,
                          batch_gt_ranks]

                outputs = []

                yield inputs, outputs
                # yield inputs

                # start a new batch
                b = 0

        except (GeneratorExit, KeyboardInterrupt):
            raise
        except Exception:
            logging.exception("Error processing image %s", dataset.get_image_info(image_id))
            consecutive_errors += 1
            if consecutive_errors > 100:   # big threshold, consecutive only
                logging.error("Too many consecutive data errors; stopping generator.")
                return  # or `break` if this is a generator function using `yield`
            continue
        else:
            # reset after a good iteration
            consecutive_errors = 0


def load_image_gt(dataset, config, image_id, augmentation=None):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: (deprecated. Use augmentation instead). If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    """
    # Load image
    # print("IMAGE ID: ", image_id)

    # Load GT data
    gt_ranks, obj_feat, shuffled_indices, sel_not_sal_obj_idx_list, \
        chosen_obj_idx_order_list, p5_feat = dataset.load_gt_rank_and_obj_feat_with_pre_proc_data(image_id)

    # # Load Object Spatial Mask
    object_roi_masks = dataset.load_object_roi_masks(image_id, sel_not_sal_obj_idx_list)

    # For 32 x 32 image size
    scale = 0.05
    padding = [(4, 4), (0, 0), (0, 0)]
    crop = None
    obj_spatial_masks = utils.resize_mask(object_roi_masks, scale, padding, crop)

    # Transpose and add dimension
    # [32, 32, N] -> [N, 32, 32, 1]
    obj_spatial_masks = np.expand_dims(np.transpose(obj_spatial_masks, [2, 0, 1]), -1)

    # Convert and make sure gt_ranks are numpy array
    gt_ranks = np.array(gt_ranks)

    # *********************** FILL REST, SHUFFLE ORDER ***********************
    # order is in salient objects then non-salient objects
    # change order to shuffled order while filling rest with 0
    batch_obj_spatial_masks = np.zeros(shape=(config.SAL_OBJ_NUM, 32, 32, 1), dtype=obj_spatial_masks.dtype)
    for k in range(len(chosen_obj_idx_order_list)):
        _idx = chosen_obj_idx_order_list[k]
        if k < len(obj_spatial_masks):
            batch_obj_spatial_masks[_idx] = obj_spatial_masks[k]

    return obj_feat, batch_obj_spatial_masks, p5_feat, gt_ranks
    # ret = dataset.load_gt_rank_and_obj_feat_with_pre_proc_data(image_id)         
    # (gt_ranks, obj_feat, shuffled_indices,
    #  sel_not_sal_obj_idx_list, chosen_obj_idx_order_list, p5_feat) = ret

    # # If any placeholder None -> skip this sample
    # if any(v is None for v in ret):
    #     return None

    # # Must have selected indices
    # if not sel_not_sal_obj_idx_list:
    #     return None

    # # ROI masks (skip if fails)
    # try:
    #     object_roi_masks = dataset.load_object_roi_masks(image_id, sel_not_sal_obj_idx_list)
    # except Exception:
    #     return None
    # if object_roi_masks is None:
    #     return None

    # # Resize masks safely
    # scale, padding, crop = 0.05, [(4, 4), (0, 0), (0, 0)], None
    # try:
    #     obj_spatial_masks = utils.resize_mask(object_roi_masks, scale, padding, crop)
    # except Exception:
    #     return None
    # if obj_spatial_masks is None or obj_spatial_masks.size == 0:
    #     return None

    # # [H,W,N] -> [N,32,32,1]
    # try:
    #     obj_spatial_masks = np.expand_dims(np.transpose(obj_spatial_masks, [2, 0, 1]), -1)
    # except Exception:
    #     return None

    # # Ensure array dtype for ranks
    # gt_ranks = np.asarray(gt_ranks, dtype=np.int32)

    # # Fill into fixed SAL_OBJ_NUM slots using chosen order, with bounds checks
    # N = config.SAL_OBJ_NUM
    # batch_obj_spatial_masks = np.zeros((N, 32, 32, 1), dtype=obj_spatial_masks.dtype)
    # for k, _idx in enumerate(chosen_obj_idx_order_list):
    #     if _idx is None:
    #         continue
    #     if not (0 <= _idx < N):
    #         continue
    #     if k >= obj_spatial_masks.shape[0]:
    #         break
    #     batch_obj_spatial_masks[_idx] = obj_spatial_masks[k]

    # return obj_feat, batch_obj_spatial_masks, p5_feat, gt_ranks


