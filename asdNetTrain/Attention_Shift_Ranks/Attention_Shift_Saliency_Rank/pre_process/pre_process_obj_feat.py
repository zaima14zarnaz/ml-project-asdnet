from model.CustomConfigs import RankModelConfig
import os
import pickle
from pre_process import Model_Obj_Feat
from pre_process import DataGenerator
from pre_process.PreProcNet import PreProcNet
from pre_process.Dataset_Object_Detection import Dataset

DATASET_ROOT = "/home/zaimaz/Desktop/research1/QAGNet/Dataset/IRSR_ASSR/"   # Change to your location
PRE_PROC_DATA_ROOT = "/home/zaimaz/Desktop/research1/QAGNet/Dataset/IRSR_ASSR/asd_extra/"    # Change to your location

if __name__ == '__main__':
    # add pre-trained weight path - backbone pre-trained on salient objects (binary, no rank)
    weight_path = "/home/zaimaz/Desktop/research1/QAGNet/asdNet/Attention_Shift_Ranks/weights/sal_seg_pretrain.h5"

    data_split = "train"

    # out_path = PRE_PROC_DATA_ROOT + "object_detection_feat/" + data_split + "/"
    out_path = PRE_PROC_DATA_ROOT + "pre_process_feat/" + data_split + "/"

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    mode = "inference"
    config = RankModelConfig()
    log_path = "logs/"

    keras_model = Model_Obj_Feat.build_obj_feat_model(config)
    model_name = "Obj_Feat_Net"

    model = PreProcNet(mode=mode, config=config, model_dir=log_path, keras_model=keras_model, model_name=model_name)

    # Load weights
    print("Loading weights ", weight_path)
    model.load_weights(weight_path, by_name=True)

    if mode == "inference":
        # ********** Create Datasets
        # obj_det_path = PRE_PROC_DATA_ROOT + "object_detection_feat/"
        obj_det_path = PRE_PROC_DATA_ROOT + "pre_process_feat/"

        # Test Dataset
        dataset = Dataset(DATASET_ROOT, data_split, obj_det_path)

        predictions = []

        num = len(dataset.img_ids)
        for i in range(num):

            image_id = dataset.img_ids[i]
            print(i + 1, " / ", num, " - ", image_id)

            input_data = DataGenerator.load_inference_data_obj_feat(dataset, image_id, config)

            result = model.detect(input_data, verbose=1)

            o_p = out_path + image_id
            with open(o_p, "wb") as f:
                pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
