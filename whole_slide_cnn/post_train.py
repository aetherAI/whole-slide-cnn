import argparse
import cv2
import json
from mpi4py import MPI
import numpy as np
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import tensorflow as tf
import tqdm
import xgboost as xgb
import yaml

from whole_slide_cnn.dataloader import MILVerboseDataloader, preprocess_input
from whole_slide_cnn.dataset import Dataset
from whole_slide_cnn.model import build_model
from whole_slide_cnn.post_train_model import build_milrnn_model, to_histogram, to_maxfeat_feature
from whole_slide_cnn.utils import ReduceLRAndRollbackOnPlateau

try:
    import horovod.tensorflow.keras as hvd
    hvd.init()
    is_rank0 = (hvd.rank() == 0)
except ImportError:
    hvd = None
    is_rank0 = True

def post_train_milrnn(
    config,
    train_dataloader,
    val_dataloader,
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

    ## Start post training
    if is_rank0:
        print("### Start post training ###")

    optimizer = tf.keras.optimizers.Adam(learning_rate=config["POST_TRAIN_INIT_LEARNING_RATE"])
    if hvd != None:
        optimizer = hvd.DistributedOptimizer(optimizer)

    post_train_model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    callbacks = []
    if hvd != None:
        callbacks += [
            hvd.callbacks.BroadcastGlobalVariablesCallback(
                0,
                device=("/cpu:0" if config["USE_HMS"] else "")
            ),
            hvd.callbacks.MetricAverageCallback(
                device=("/cpu:0" if config["USE_HMS"] else ""),
            ),
        ]
    callbacks.append(
        ReduceLRAndRollbackOnPlateau(
            filepath=config["POST_TRAIN_MODEL_PATH"],
            factor=config["POST_TRAIN_REDUCE_LR_FACTOR"],
            min_lr=1e-8,
            monitor="val_loss",
            patience=config["POST_TRAIN_REDUCE_LR_PATIENCE"],
            verbose=(1 if is_rank0 else 0),
        )
    )
    if is_rank0:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=config["POST_TRAIN_MODEL_PATH"],
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=(1 if is_rank0 else 0),
            )
        )

    post_train_model.fit(
        train_dataloader,
        workers=0, # MIL dataloader should be in the main thread
        max_queue_size=1,
        use_multiprocessing=False,
        epochs=config["POST_TRAIN_EPOCHS"],
        steps_per_epoch=config["POST_TRAIN_NUM_UPDATES_PER_EPOCH"],
        validation_data=val_dataloader,
        callbacks=callbacks,
        shuffle=False, # Shuffling is already done in dataloader
        verbose=(1 if is_rank0 else 0),
    )

def post_train_hou(
    config,
    train_dataloader,
    val_dataloader,
):
    method = config["POST_TRAIN_METHOD"]

    ## Get and preprocess training data to class histograms
    if is_rank0:
        print("### Get and preprocess training data ###")

    histogram_list, pred_list = to_histogram(train_dataloader, is_rank0)

    # Gather training data to rank 0
    if hvd != None:
        histogram_list = histogram_list.tolist()
        pred_list = pred_list.tolist()

        all_histogram_list = MPI.COMM_WORLD.gather(
            histogram_list,
            root=0
        )
        all_pred_list = MPI.COMM_WORLD.gather(
            pred_list,
            root=0
        )

        if is_rank0:
            all_histogram_list = np.concatenate(np.array(all_histogram_list), axis=0)
            all_pred_list = np.concatenate(np.array(all_pred_list), axis=0)

        histogram_list = all_histogram_list
        pred_list = all_pred_list

    ## Train the post_train model
    if is_rank0:
        print("### Training the post-train model ###")

        if method == "svm":
            post_train_model = SVC(kernel="rbf", probability=True)
        elif method == "lr":
            post_train_model = LogisticRegression(max_iter=1000)
        else:
            raise NotImplementedError()

        post_train_model.fit(histogram_list, pred_list)
        accuracy = post_train_model.score(histogram_list, pred_list)
        print("Train Accuracy: {}".format(accuracy))
        with open(config["POST_TRAIN_MODEL_PATH"], "wb") as f:
            pickle.dump(post_train_model, f)
        print("Post-train model saved at {}.".format(config["POST_TRAIN_MODEL_PATH"]))
    

def post_train_maxfeat_rf(
    config,
    train_dataloader,
    val_dataloader,
):
    ## Get and preprocess training data to maxfeat
    if is_rank0:
        print("### Get and preprocess training data ###")

    maxfeat_list, pred_list = to_maxfeat_feature(train_dataloader, is_rank0)

    # Gather training data to rank 0
    if hvd != None:
        maxfeat_list = maxfeat_list.tolist()
        pred_list = pred_list.tolist()

        all_maxfeat_list = MPI.COMM_WORLD.gather(
            maxfeat_list,
            root=0
        )
        all_pred_list = MPI.COMM_WORLD.gather(
            pred_list,
            root=0
        )

        if is_rank0:
            all_maxfeat_list = np.concatenate(np.array(all_maxfeat_list), axis=0)
            all_pred_list = np.concatenate(np.array(all_pred_list), axis=0)

        maxfeat_list = all_maxfeat_list
        pred_list = all_pred_list

    ## Train the post_train model
    if is_rank0:
        print("### Training the post-train model ###")

        post_train_model = xgb.XGBRFClassifier()

        post_train_model.fit(maxfeat_list, pred_list)
        accuracy = post_train_model.score(maxfeat_list, pred_list)
        print("Train Accuracy: {}".format(accuracy))
        with open(config["POST_TRAIN_MODEL_PATH"], "wb") as f:
            pickle.dump(post_train_model, f)
        print("Post-train model saved at {}.".format(config["POST_TRAIN_MODEL_PATH"]))
    

if __name__ == "__main__":
    ## Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="YAML config file from train_configs/",
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

    if not config["USE_MIL"]:
        raise ValueError("Post training is only available for MIL models")

    if config["POST_TRAIN_METHOD"] == None or config["POST_TRAIN_METHOD"] == "":
        raise ValueError("No post train method is specified.")

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
    train_dataset = Dataset(
        csv_path=config["TRAIN_CSV_PATH"],
        slide_dir=config["SLIDE_DIR"],
        slide_file_extension=config["SLIDE_FILE_EXTENSION"],
        target_size=config["INPUT_SIZE"][: 2],
        resize_ratio=config["RESIZE_RATIO"],
        slide_reader=config["SLIDE_READER"],
        snapshot_path=(config["DEBUG_PATH"] if is_rank0 else None),
    )
    if is_rank0:
        print("Train dataset contains {} slides.".format(len(train_dataset)))
    val_dataset = Dataset(
        csv_path=config["VAL_CSV_PATH"],
        slide_dir=config["SLIDE_DIR"],
        slide_file_extension=config["SLIDE_FILE_EXTENSION"],
        target_size=config["INPUT_SIZE"][: 2],
        resize_ratio=config["RESIZE_RATIO"],
        slide_reader=config["SLIDE_READER"],
        snapshot_path=(config["DEBUG_PATH"] if is_rank0 else None),
    )
    if is_rank0:
        print("Validation dataset contains {} slides.".format(len(val_dataset)))

    ## Initialize the primary model
    if is_rank0:
        print("### Initializing the primary model ###")

    model = build_model(
        input_shape=(mil_patch_size + [config["INPUT_SIZE"][2]]),
        n_classes=config["NUM_CLASSES"],
        backbone=config["MODEL"],
        pool_use=config["POOL_USE"],
        use_mixed_precision=config["USE_MIXED_PRECISION"],
        batch_size=config["BATCH_SIZE"],
        use_huge_layer_wrapper=False,
        for_post_train=True,
    )
    if is_rank0:
        model.summary()

    model.load_weights(config["MODEL_PATH"])
    if is_rank0:
        print("The weight file {} is loaded.".format(config["MODEL_PATH"]))

    ## Initialize the dataloader
    if is_rank0:
        print("### Initializing dataloaders ###")
    train_dataloader = MILVerboseDataloader(
        dataset=train_dataset,
        augment=False,
        shuffle=True,
        num_classes=config["NUM_CLASSES"],
        batch_size=config["BATCH_SIZE"],
        snapshot_path=(config["DEBUG_PATH"] if is_rank0 else None),
        hvd=hvd,
        mil_model=model,
        mil_patch_size=mil_patch_size,
        mil_infer_batch_size=config["MIL_INFER_BATCH_SIZE"],
    )
    print(
        "Train dataloader contains {} slides on rank {}. The first one is {}.".format(
            len(train_dataloader),
            hvd.rank() if hvd != None else 0,
            train_dataloader.get_slide_path(0),
        )
    )
    val_dataloader = MILVerboseDataloader(
        dataset=val_dataset,
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
    print(
        "Validation dataloader contains {} slides on rank {}. The first one is {}.".format(
            len(val_dataloader),
            hvd.rank() if hvd != None else 0,
            val_dataloader.get_slide_path(0),
        )
    )

    if config["POST_TRAIN_METHOD"] == "milrnn":
        post_train_milrnn(
            config=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            mil_patch_size=mil_patch_size,
        )
    elif config["POST_TRAIN_METHOD"] in ["svm", "lr"]:
        post_train_hou(
            config=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
        )
    elif config["POST_TRAIN_METHOD"] == "maxfeat_rf":
        post_train_maxfeat_rf(
            config=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
        )
    else:
        raise NotImplementedError("Unsupported post train method.")

