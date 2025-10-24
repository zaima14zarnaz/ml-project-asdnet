from Dataset import Dataset
from model.CustomConfigs import RankModelConfig
from model.ASRNet import ASRNet
import DataGenerator
from model import Model_SAM_SMM

DATASET_ROOT = "/home/zaimaz/Desktop/research1/QAGNet/Dataset/IRSR_ASSR/"   # Change to your location
PRE_PROC_DATA_ROOT = "/home/zaimaz/Desktop/research1/QAGNet/Dataset/IRSR_ASSR/asd_extra/"    # Change to your location


if __name__ == '__main__':
    # weight_path = "/home/zaimaz/Desktop/research1/QAGNet/asdNet/Attention_Shift_Ranks/Attention_Shift_Saliency_Rank/logs/obj_sal_seg_mask_config20250911T2245/sos_net_obj_sal_seg_mask_config_0040.h5"    # add pre-trained weight path
    weight_path = "../weights/ASRNet_model_weights.h5"

    command = "train"
    config = RankModelConfig()
    log_path = "logs/"
    mode = "training"

    print("Loading Rank Model")
    keras_model = Model_SAM_SMM.build_saliency_rank_model(config, mode)
    model_name = "Rank_Model_SAM_SMM"
    model = ASRNet(mode=mode, config=config, model_dir=log_path, keras_model=keras_model, model_name=model_name)

    # Load weights
    # print("Loading weights ", weight_path)
    # model.load_weights(weight_path, by_name=True)

    # Train/Evaluate Model
    if command == "train":
        print("Start Training...")

        # ********** Create Datasets
        # Train Dataset
        train_dataset = Dataset(DATASET_ROOT, PRE_PROC_DATA_ROOT, "train")

        # Val Dataset
        val_dataset = Dataset(DATASET_ROOT, PRE_PROC_DATA_ROOT, "val")

        # ********** Parameters
        # Image Augmentation
        # Right/Left flip 50% of the time
        # augmentation = imgaug.augmenters.Fliplr(0.5)
        augmentation = None

        # ********** Create Data generators
        train_generator = DataGenerator.data_generator(train_dataset, config, shuffle=True,
                                                       augmentation=augmentation,
                                                       batch_size=config.BATCH_NUM)
        val_generator = DataGenerator.data_generator(val_dataset, config, shuffle=True,
                                                     batch_size=config.BATCH_NUM)

        # ********** Training  **********
        model.train(train_generator, val_generator,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='all')
