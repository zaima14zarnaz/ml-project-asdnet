from model.CustomConfigs import RankModelConfig
import os
import pickle
from pre_process.Dataset_Object_Detection import Dataset
from fpn_network.FPN import FPN

# DATASET_ROOT = "/home/zaimaz/Desktop/research1/QAGNet/Dataset/IRSR_ASSR/"   # Change to your location
# PRE_PROC_DATA_ROOT = "/home/zaimaz/Desktop/research1/QAGNet/Dataset/IRSR_ASSR/asd_extra/"    # Change to your location
# DATASET_ROOT = "/home/zaimaz/Desktop/research1/QAGNet/Dataset/ASSR/"   # Change to your location
# PRE_PROC_DATA_ROOT = "/home/zaimaz/Desktop/research1/QAGNet/Dataset/ASSR/ASSR_data/"    # Change to your location
DATASET_ROOT = "/home/zaimaz/Desktop/research1/QAGNet/Dataset/SIFR_ASSR/"   # Change to your location
PRE_PROC_DATA_ROOT = "/home/zaimaz/Desktop/research1/QAGNet/Dataset/SIFR_ASSR/asd_extra/"    # Change to your location




if __name__ == '__main__':
    # add pre-trained weight path - backbone pre-trained on salient objects (binary, no rank)
    weight_path = "/home/zaimaz/Desktop/research1/QAGNet/asdNetTest/Attention_Shift_Ranks/Attention_Shift_Saliency_Rank/weights/sal_seg_pretrain(1).h5"

    out_path = PRE_PROC_DATA_ROOT + "object_detection_feat/"

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    mode = "inference"
    config = RankModelConfig()
    log_path = "logs/"

    model = FPN(mode=mode, config=config, model_dir=log_path)

    # Load weights
    print("Loading weights ", weight_path)
    model.load_weights(weight_path, by_name=True)
    data_split = "test"

    if mode == "inference":
        # Test Dataset
        dataset = Dataset(DATASET_ROOT, data_split)

        predictions = []

        num = len(dataset.img_ids)
        for i in range(num):

            image_id = dataset.img_ids[i]
            print("\n", i + 1, " / ", num, " - ", image_id)

            image = dataset.load_image(image_id)

            result = model.detect([image], verbose=1)
            pr = result[0]

            rois = pr["rois"].tolist()
            class_ids = pr["class_ids"].tolist()
            scores = pr["scores"].tolist()

            res = {}
            res["image_id"] = image_id
            res["rois"] = rois
            res["class_ids"] = class_ids
            res["scores"] = scores
            predictions.append(res)

        o_p = out_path + f"object_detection_{data_split}_images.pkl"
        with open(o_p, "wb") as f:
            pickle.dump(predictions, f, pickle.HIGHEST_PROTOCOL)



