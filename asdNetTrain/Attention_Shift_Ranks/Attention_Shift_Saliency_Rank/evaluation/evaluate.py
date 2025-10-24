from evaluation.DatasetTest import DatasetTest
from sklearn.metrics import mean_absolute_error
import cv2
import numpy as np
import pickle
import os
import utils
import scipy.stats as sc

WIDTH = 640
HEIGHT = 480

# Percentage of object pixels having predicted saliency value to consider as salient object
# for cases where object segments overlap each other
SEG_THRESHOLD = .5


def load_saliency_map(path):
    # Load mask
    sal_map = cv2.imread(path, 1)
    if sal_map is None:
        path = path
        return None
    sal_map = sal_map.astype(np.float32)

    # Need only one channel
    sal_map = sal_map[:, :, 0]

    # Normalize to 0-1
    sal_map /= 255.0

    return sal_map


def eval_mae(dataset, map_path):
    print("Calculating MAE...")

    mae_list = []

    num = len(dataset.img_ids)
    for i in range(num):
        image_id = dataset.img_ids[i]

        p = map_path + image_id + ".png"
        # print(p)

        pred_mask = load_saliency_map(p)

        gt_mask = dataset.load_gt_mask(image_id)
        if gt_mask is None:
            continue

        # Flatten masks
        gt_mask = gt_mask.flatten()
        pred_mask = pred_mask.flatten()

        mae = mean_absolute_error(gt_mask, pred_mask)

        mae_list.append(mae)

    print("\n")
    avg_mae = sum(mae_list) / len(mae_list)
    print("Average MAE Images = ", avg_mae)


def eval_mae_binary_mask(dataset, map_path):
    print("Calculating MAE (Binary Saliency)...")

    mae_list = []

    num = len(dataset.img_ids)
    for i in range(num):
        image_id = dataset.img_ids[i]

        p = map_path + image_id + ".png"
        pred_mask = load_saliency_map(p)

        gt_mask = dataset.load_gt_mask(image_id)

        # Convert masks to binary
        pred_mask[pred_mask > 0] = 1
        gt_mask[gt_mask > 0] = 1

        # Flatten masks
        gt_mask = gt_mask.flatten()
        pred_mask = pred_mask.flatten()

        mae = mean_absolute_error(gt_mask, pred_mask)

        mae_list.append(mae)

    print("\n")
    avg_mae = sum(mae_list) / len(mae_list)
    print("Average MAE Images (Binary Masks) = ", avg_mae)


def calculate_spr(dataset, model_pred_data_path, out_path):
    print("Calculating SOR and SA-SOR...")

    gt_rank_order = dataset.gt_rank_orders
    spr_data = []

    same_score = 0

    num = min(len(dataset.img_ids), len(dataset.sal_obj_idx_list))
    print(f'length of image id: {len(dataset.img_ids)}')
    print(f'length of image id: {len(dataset.sal_obj_idx_list)}')
    for i in range(num):
        image_id = dataset.img_ids[i]
        print("\n")
        print(i + 1, "/", num, "-", image_id)

        sal_obj_idx = dataset.sal_obj_idx_list[i]
        N = len(sal_obj_idx)

        obj_seg = dataset.obj_seg[i]
        instance_masks = []
        instance_pix_count = []

        for s_i in range(len(sal_obj_idx)):
            sal_idx = sal_obj_idx[s_i]

            if sal_idx >= len(obj_seg):
                print(f"[Warning] sal_idx {sal_idx} out of bounds for obj_seg with length {len(obj_seg)} — skipping.")
                continue

            seg = obj_seg[sal_idx]
            mask = utils.get_obj_mask(seg, HEIGHT, WIDTH)
            pix_count = mask.sum()

            instance_masks.append(mask)
            instance_pix_count.append(pix_count)

        pred_data_path = model_pred_data_path + image_id + ".png"
        # print(pred_data_path)
        pred_sal_map = cv2.imread(pred_data_path)[:, :, 0]
        if pred_sal_map.shape != (HEIGHT, WIDTH):
            print(f"[Warning] Resizing predicted saliency map from {pred_sal_map.shape} to ({HEIGHT},{WIDTH})")
            pred_sal_map = cv2.resize(pred_sal_map, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)

        print(f'Predicted saliency map shape: {pred_sal_map.shape} at data path: {pred_data_path}')

        pred_ranks = []

        for s_i in range(len(instance_masks)):
            gt_seg_mask = instance_masks[s_i]
            gt_pix_count = instance_pix_count[s_i]
            pred_seg = np.where(gt_seg_mask == 1, pred_sal_map, 0)

            pred_pix_loc = np.where(pred_seg > 0)
            pred_pix_num = len(pred_pix_loc[0])
            r = 0

            print(f'pred_pix_num: {pred_pix_num}, int(gt_pix_count * SEG_THRESHOLD): {int(gt_pix_count * SEG_THRESHOLD)}')

            if pred_pix_num > int(gt_pix_count * SEG_THRESHOLD):
                vals = pred_seg[pred_pix_loc[0], pred_pix_loc[1]]
                mean_val = vals.mean()
                print(f"[Segment {s_i}] Average gray value of predicted segment: {mean_val:.2f}")

                res = sc.mode(vals, axis=None, keepdims=False)
                mode = res.mode.item() if res.mode.size else None
                r = mode


            pred_ranks.append(r)

        gt_rank_order_list = gt_rank_order[i]
        gt_ranks = []
        valid_pix_counts = []

        for j in range(N):
            s_idx = sal_obj_idx[j]
            gt_r = gt_rank_order_list[s_idx]
            gt_ranks.append(gt_r)
            if j < len(instance_pix_count):  # match pixel counts to gt indices
                valid_pix_counts.append(instance_pix_count[j])

        # Trim to common length
        # print(gt_ranks)
        # print(pred_ranks)
        min_len = min(len(gt_ranks), len(pred_ranks))
        gt_ranks = gt_ranks[:min_len]
        pred_ranks = pred_ranks[:min_len]
        valid_pix_counts = valid_pix_counts[:min_len]

        gt_ranks, pred_ranks, use_indices_list = \
            utils.get_usable_salient_objects_agreed(gt_ranks, pred_ranks)

        valid_pix_counts = [valid_pix_counts[j] for j in use_indices_list]

        spr = None
        sa_spr = None

        if len(gt_ranks) > 1:
            spr = sc.spearmanr(gt_ranks, pred_ranks).correlation
            try:
                print("valid_pix_counts:", valid_pix_counts)
                sa_spr = sc.spearmanr(gt_ranks, pred_ranks, alternative='two-sided', nan_policy='omit',
                                      axis=0, weights=valid_pix_counts).correlation
            except TypeError:
                # Fallback if `spearmanr(..., weights=...)` not supported in current scipy version
                print("[Warning] Your scipy version does not support weighted Spearman. Using unweighted SOR.")
                sa_spr = spr
        elif len(gt_ranks) == 1:
            spr = sa_spr = 1
        
        if spr == sa_spr:
            same_score += 1

        d = [image_id, spr, sa_spr, use_indices_list]
        spr_data.append(d)
        
    print(f'{same_score}/{len(spr_data)} cases, sa and sa_sor scores are the same.')
    with open(out_path, "wb") as f:
        pickle.dump(spr_data, f)



def extract_spr_value(data_list):
    use_idx_list = []
    spr = []
    for i in range(len(data_list)):
        s = data_list[i][1]

        if s == 1:
            spr.append(s)
            use_idx_list.append(i)
        elif s and not np.isnan(s[0]):
            spr.append(s[0])
            use_idx_list.append(i)

    return spr, use_idx_list


def cal_avg_spr(data_list):
    data_list = [x for x in data_list if x is not None and not np.isnan(x)]
    return np.mean(data_list) if data_list else float('nan')


def get_norm_spr(spr_value):
    #       m - r_min
    # m -> ---------------- x (t_max - t_min) + t_min
    #       r_max - r_min
    #
    # m = measure value
    # r_min = min range of measurement
    # r_max = max range of measurement
    # t_min = min range of desired scale
    # t_max = max range of desired scale

    r_min = -1
    r_max = 1

    norm_spr = (spr_value - r_min) / (r_max - r_min)

    return norm_spr


def eval_spr(spr_data_path):
    with open(spr_data_path, "rb") as f:
        spr_all_data = pickle.load(f)

    # Separate out SOR and SA-SOR scores
    sor_data = []
    sa_sor_data = []

    for item in spr_all_data:
        image_id, sor, sa_sor, _ = item

        if sor is not None:
            sor_data.append(sor)
        if sa_sor is not None:
            sa_sor_data.append(sa_sor)

    # Separate positive and negative SORs
    pos_sor = [v for v in sor_data if v > 0]
    neg_sor = [v for v in sor_data if v <= 0]

    pos_sa_sor = [v for v in sa_sor_data if v > 0]
    neg_sa_sor = [v for v in sa_sor_data if v <= 0]

    print("SOR values (sample):", sor_data[:10])
    print("SA-SOR values (sample):", sa_sor_data[:10])
    print("Number of valid SORs:", len([x for x in sor_data if not np.isnan(x)]))
    print("Number of valid SA-SORs:", len([x for x in sa_sor_data if not np.isnan(x)]))


    print("---- SOR (Spearman Rank) ----")
    print("Positive SOR: ", len(pos_sor))
    print("Negative SOR: ", len(neg_sor))
    print("Average SOR: ", cal_avg_spr(sor_data))
    print("Normalized Average SOR: ", get_norm_spr(cal_avg_spr(sor_data)))

    print("\n---- SA-SOR (Size-Aware SOR) ----")
    print("Positive SA-SOR: ", len(pos_sa_sor))
    print("Negative SA-SOR: ", len(neg_sa_sor))
    print("Average SA-SOR: ", cal_avg_spr(sa_sor_data))
    print("Normalized Average SA-SOR: ", get_norm_spr(cal_avg_spr(sa_sor_data)))

    print("\n----------------------------------------------------------")
    print("Data path: ", spr_data_path)
    print(len(sor_data), "/", len(spr_all_data), " - ", (len(spr_all_data) - len(sor_data)), "Images Not used")



if __name__ == '__main__':
    print("Evaluate")

    DATASET_ROOT = "/home/zaimaz/Desktop/research1/QAGNet/Dataset/IRSR_ASSR/"   # Change to your location
    PRE_PROC_DATA_ROOT = "/home/zaimaz/Desktop/research1/QAGNet/Dataset/IRSR_ASSR/asd_extra/"    # Change to your location


    data_split = "test"
    dataset = DatasetTest(DATASET_ROOT, PRE_PROC_DATA_ROOT, data_split, eval_spr=True)

    ####################################################
    map_path = "../saliency_maps/"

    # # Calculate MAE
    eval_mae(dataset, map_path)
    eval_mae_binary_mask(dataset, map_path)

    ####################################################
    out_root = "../spr_data/"
    out_path = out_root + "spr_data"
    if not os.path.exists(out_root):
        os.makedirs(out_root)

    # Calculate SOR
    calculate_spr(dataset, map_path, out_path)

    eval_spr(out_path)
