import argparse
import cv2
import json
from mpi4py import MPI
import numpy as np
import os
import pickle
import tensorflow as tf
import yaml

from whole_slide_cnn.dataloader import (
    WholeSlideDataloader, 
    MILDataloader, 
    MILVerboseDataloader, 
    preprocess_input,
)
from whole_slide_cnn.dataset import Dataset
from whole_slide_cnn.model import build_model
from whole_slide_cnn.post_train_model import build_milrnn_model, to_histogram, to_maxfeat_feature
from whole_slide_cnn.utils import TimeRecorder, get_cam, get_mil_pred_map, draw_pred_map

try:
    import horovod.tensorflow.keras as hvd
    hvd.init()
    is_rank0 = (hvd.rank() == 0)
except ImportError:
    hvd = None
    is_rank0 = True

def test_post_train_hou(
    config,
    test_dataloader,
):
    method = config["POST_TRAIN_METHOD"]

    ## Get and preprocess training data to class histograms
    if is_rank0:
        print("### Get and preprocess testing data ###")

    histogram_list, _ = to_histogram(test_dataloader, is_rank0)

    ## Test the post_train model
    if is_rank0:
        print("### Testing the post-train model ###")

    with open(config["POST_TRAIN_MODEL_PATH"], "rb") as f:
        post_train_model = pickle.load(f)

    pred_list = post_train_model.predict_proba(histogram_list)

    return pred_list

def test_post_train_milrnn(
    config,
    test_dataloader,
    mil_patch_size,
):
    ## Initialize the post-train model
    if is_rank0:
        print("### Initializing the post-train model ###")

    emb_dim = {
        "resnet34": 512,
        "fixup_resnet34": 512,
        "fixup_resnet50": 2048,
    }[config["MODEL"]]
    input_shape = [
        config["INPUT_SIZE"][0] // mil_patch_size[0],
        config["INPUT_SIZE"][1] // mil_patch_size[1],
    ]

    post_train_model = build_milrnn_model(
        input_shape=input_shape,
        emb_dim=emb_dim,
        n_classes=config["NUM_CLASSES"],
    )
    if is_rank0:
        post_train_model.summary()

    post_train_model.load_weights(config["POST_TRAIN_MODEL_PATH"])

    ## Test the post_train model
    if is_rank0:
        print("### Testing the post-train model ###")

    pred_list = post_train_model.predict(
        test_dataloader,
        workers=0, # MIL dataloader should be in the main thread
        verbose=(1 if is_rank0 else 0),
    )

    return pred_list
    
def test_post_train_maxfeat_rf(
    config,
    test_dataloader,
):
    ## Get and preprocess training data to maxfeat
    if is_rank0:
        print("### Get and preprocess training data ###")

    maxfeat_list, _ = to_maxfeat_feature(test_dataloader, is_rank0) 

    ## Test the post_train model
    if is_rank0:
        print("### Testing the post-train model ###")

    with open(config["POST_TRAIN_MODEL_PATH"], "rb") as f:
        post_train_model = pickle.load(f)

    pred_list = post_train_model.predict_proba(maxfeat_list)

    return pred_list

if __name__ == "__main__":
    ## Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="sample_configs/config_whole_slide.yaml",
    )
    args = parser.parse_args()

    ## Read the config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # Replace ${RESULT_DIR}
    for key in config:
        if not isinstance(config[key], str):
            continue
        config[key] = config[key].replace("${RESULT_DIR}", config["RESULT_DIR"])

    if is_rank0:
        print("### Config ###")
        print(json.dumps(config, indent=4))

    if int(config["USE_HMS"]) + int(config["USE_MIL"]) > 1:
        raise ValueError("Exact one of USE_HMS and USE_MIL should be set True.")

    if (
        "POST_TRAIN_METHOD" not in config or
        config["POST_TRAIN_METHOD"] == None or 
        config["POST_TRAIN_METHOD"] == ""
    ):
        use_post_train = False
    else:
        use_post_train = True

    if use_post_train:
        if (
            "POST_TRAIN_MIL_PATCH_SIZE" not in config or
            config["POST_TRAIN_MIL_PATCH_SIZE"] == None 
        ):
            mil_patch_size = config["MIL_PATCH_SIZE"]
        else:
            mil_patch_size = config["POST_TRAIN_MIL_PATCH_SIZE"]

    ## Initialize tensorflow_huge_model_support (HMS) and horovod if required.
    config_proto = tf.ConfigProto()
    if hvd != None:
        config_proto.gpu_options.visible_device_list = str(hvd.local_rank())
    config_proto.gpu_options.per_process_gpu_memory_fraction = 10.0
    sess = tf.Session(config=config_proto)
    tf.keras.backend.set_session(sess)

    ## Initialize datasets
    if is_rank0:
        print("### Initializing datasets ###")
    test_dataset = Dataset(
        csv_path=config["TEST_CSV_PATH"],
        slide_dir=config["SLIDE_DIR"],
        slide_file_extension=config["SLIDE_FILE_EXTENSION"],
        target_size=config["INPUT_SIZE"][: 2],
        resize_ratio=config["RESIZE_RATIO"],
        slide_reader=config["SLIDE_READER"],
        use_tcga_vahadane=(False if "USE_TCGA_VAHADANE" not in config else config["USE_TCGA_VAHADANE"]),
        snapshot_path=(config["DEBUG_PATH"] if is_rank0 else None),
    )
    if is_rank0:
        print("Test dataset contains {} slides.".format(len(test_dataset)))

    ## Initialize the model
    if is_rank0:
        print("### Initializing the model ###")
    if config["USE_MIL"]:
        model = build_model(
            input_shape=(mil_patch_size + [config["INPUT_SIZE"][2]]),
            n_classes=config["NUM_CLASSES"],
            backbone=config["MODEL"],
            pool_use=config["POOL_USE"],
            use_mixed_precision=config["USE_MIXED_PRECISION"],
            batch_size=config["BATCH_SIZE"],
            use_huge_layer_wrapper=False,
            for_post_train=use_post_train,
        )
    else:
        model = build_model(
            input_shape=config["INPUT_SIZE"],
            n_classes=config["NUM_CLASSES"],
            backbone=config["MODEL"],
            pool_use=config["POOL_USE"],
            use_mixed_precision=config["USE_MIXED_PRECISION"],
            batch_size=config["BATCH_SIZE"],
            use_huge_layer_wrapper=config["USE_HMS"],
        )
    if is_rank0:
        model.summary()

    model.load_weights(config["MODEL_PATH"])
    if is_rank0:
        print("The weight file {} is loaded.".format(config["MODEL_PATH"]))

    ## Initialize dataloaders specific to either whole-slide training or MIL.
    if is_rank0:
        print("### Initializing dataloaders ###")
    if config["USE_MIL"]:
        if use_post_train:
            test_dataloader = MILVerboseDataloader(
                dataset=test_dataset,
                augment=False,
                shuffle=False,
                num_classes=config["NUM_CLASSES"],
                batch_size=config["BATCH_SIZE"],
                snapshot_path=(config["DEBUG_PATH"] if is_rank0 else None),
                hvd=hvd,
                mil_model=model,
                mil_patch_size=mil_patch_size,
                mil_infer_batch_size=config["MIL_INFER_BATCH_SIZE"],
            )
        else:
            test_dataloader = MILDataloader(
                dataset=test_dataset,
                augment=False,
                shuffle=False,
                num_classes=config["NUM_CLASSES"],
                batch_size=config["BATCH_SIZE"],
                snapshot_path=(config["DEBUG_PATH"] if is_rank0 else None),
                hvd=hvd,
                mil_model=model,
                mil_patch_size=mil_patch_size,
                mil_infer_batch_size=config["MIL_INFER_BATCH_SIZE"],
                mil_use_em=False,
                mil_k=1,
            )
    else:
        test_dataloader = WholeSlideDataloader(
            dataset=test_dataset,
            augment=False,
            shuffle=False,
            num_classes=config["NUM_CLASSES"],
            batch_size=config["BATCH_SIZE"],
            snapshot_path=(config["DEBUG_PATH"] if is_rank0 else None),
            hvd=hvd,
        )
    print(
        "Test dataloader contains {} slides on rank {}. The first one is {}.".format(
            len(test_dataloader),
            hvd.rank() if hvd != None else 0,
            test_dataloader.get_slide_path(0),
        )
    )

    ## Start testing
    if is_rank0:
        print("### Start testing ###")

    callbacks = []
    if is_rank0:
        callbacks.append(
            TimeRecorder(
                filepath=config["TEST_TIME_RECORD_PATH"],
            )
        )

    if config["USE_MIL"]:
        if use_post_train:
            if config["POST_TRAIN_METHOD"] == "milrnn":
                y_pred_list = test_post_train_milrnn(
                    config=config,
                    test_dataloader=test_dataloader,
                    mil_patch_size=mil_patch_size,
                )
            elif config["POST_TRAIN_METHOD"] in ["svm", "lr"]:
                y_pred_list = test_post_train_hou(
                    config=config,
                    test_dataloader=test_dataloader,
                )
            elif config["POST_TRAIN_METHOD"] == "maxfeat_rf":
                y_pred_list = test_post_train_maxfeat_rf(
                    config=config,
                    test_dataloader=test_dataloader,
                )
            else:
                raise NotImplementedError()
        else:
            y_pred_list = model.predict(
                test_dataloader,
                workers=0, # MIL dataloader should be in the main thread
                max_queue_size=1,
                use_multiprocessing=False,
                callbacks=callbacks,
                verbose=(1 if is_rank0 else 0),
            )
    else:
        y_pred_list = model.predict(
            test_dataloader,
            workers=1,
            max_queue_size=1,
            use_multiprocessing=False,
            callbacks=callbacks,
            verbose=(1 if is_rank0 else 0),
        )

    test_result_list = []
    for idx, y_pred in enumerate(y_pred_list):
        slide_path = test_dataloader.get_slide_path(idx)
        y_true = test_dataloader.get_y_true(idx)
        y_pred = y_pred.tolist()

        test_result = {
            "slide_path": slide_path,
            "y_true": y_true,
            "y_pred": y_pred,
        }
        test_result_list.append(test_result)
    
    if hvd != None:
        all_test_result_list = MPI.COMM_WORLD.gather(test_result_list, root=0)
    if is_rank0:
        num_results = 0
        for x in all_test_result_list:
            num_results += len(x)

        flat_test_result_list = []
        num_workers = MPI.COMM_WORLD.Get_size()
        for i in range(num_results):
            flat_test_result_list.append(all_test_result_list[i % num_workers][i // num_workers])

        with open(config["TEST_RESULT_PATH"], "w") as f:
            json.dump(flat_test_result_list, f, indent=4)

        print("Testing result saved as {}.".format(config["TEST_RESULT_PATH"]))

    # Start visualizing
    if config["ENABLE_VIZ"]:
        for idx, y_pred in enumerate(y_pred_list):
            slide_basename = os.path.basename(test_dataloader.get_slide_path(idx))
            y_true = test_dataloader.get_y_true(idx)
            dataset_idx = test_dataloader.get_dataset_idx(idx)
            preprocessed_input_image = preprocess_input(test_dataset[dataset_idx][0])

            if config["USE_MIL"]:
                pred_map = get_mil_pred_map(
                    input_image=preprocessed_input_image,
                    model=model,
                    class_select=y_true,
                )
            else:
                pred_map = get_cam(
                    input_image=preprocessed_input_image,
                    model=model,
                    class_select=y_true,
                )
            
            heatmap_bgra = draw_pred_map(
                pred_map,
                size=config["VIZ_SIZE"],
            )

            os.makedirs(config["VIZ_FOLDER"], exist_ok=True)
            viz_path = os.path.join(
                config["VIZ_FOLDER"],
                "{}_class{}.tiff".format(slide_basename, y_true),
            )
            cv2.imwrite(
                viz_path,
                heatmap_bgra,
            )
            print("Visualization result saved as {}.".format(viz_path))

