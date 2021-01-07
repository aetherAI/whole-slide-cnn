import argparse
import json
import os
import shutil
import tensorflow as tf
import yaml

from whole_slide_cnn.dataloader import WholeSlideDataloader, MILDataloader
from whole_slide_cnn.dataset import Dataset
from whole_slide_cnn.model import build_model
from whole_slide_cnn.utils import ReduceLRAndRollbackOnPlateau, TimeRecorder

try:
    import horovod.tensorflow.keras as hvd
    hvd.init()
    is_rank0 = (hvd.rank() == 0)
except ImportError:
    hvd = None
    is_rank0 = True

try:
    from tensorflow_huge_model_support.tf_keras import init as hms_init, HMSTFKerasCallback
except ImportError:
    hms_init = None
    HMSTFKerasCallback = None

if __name__ == "__main__":
    ## Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="sample_configs/config_whole_slide.yaml",
    )
    parser.add_argument(
        "--continue_mode",
        type=bool,
        nargs='?',
        const=True,
        default=False,
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

    if args.continue_mode:
        if is_rank0:
            print("!!! CONTINUE MODE !!!")
        config["LOAD_MODEL_BEFORE_TRAIN"] = True

    if is_rank0:
        print("### Config ###")
        print(json.dumps(config, indent=4))

        try:
            shutil.copyfile(args.config, config["CONFIG_RECORD_PATH"])
        except:
            pass

    if int(config["USE_HMS"]) + int(config["USE_MIL"]) > 1:
        raise ValueError("Exact one of USE_HMS and USE_MIL should be set True.")

    os.makedirs(config["RESULT_DIR"], exist_ok=True)

    ## Initialize tensorflow_huge_model_support (HMS) and horovod if required.
    if config["USE_HMS"]:
        assert hms_init != None, "Please install tensorflow_huge_model_support first."
        hms_init(hvd=hvd)
    else:
        config_proto = tf.ConfigProto()
        if hvd != None:
            config_proto.gpu_options.visible_device_list = str(hvd.local_rank())
        config_proto.gpu_options.per_process_gpu_memory_fraction = 100.0
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
        print("Training dataset contains {} slides.".format(len(train_dataset)))
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

    ## Initialize the model
    if is_rank0:
        print("### Initializing the model ###")
    if config["USE_MIL"]:
        model = build_model(
            input_shape=(config["MIL_PATCH_SIZE"] + [config["INPUT_SIZE"][2]]),
            n_classes=config["NUM_CLASSES"],
            backbone=config["MODEL"],
            pool_use=config["POOL_USE"],
            use_mixed_precision=config["USE_MIXED_PRECISION"],
            batch_size=config["BATCH_SIZE"],
            use_huge_layer_wrapper=False,
        )
    else:
        model = build_model(
            input_shape=config["INPUT_SIZE"],
            n_classes=config["NUM_CLASSES"],
            backbone=config["MODEL"],
            pool_use=config["POOL_USE"],
            use_mixed_precision=config["USE_MIXED_PRECISION"],
            batch_size=config["BATCH_SIZE"],
            use_huge_layer_wrapper=True,
        )
    if is_rank0:
        model.summary()

    if config["LOAD_MODEL_BEFORE_TRAIN"]:
        model.load_weights(config["MODEL_PATH"])
        if is_rank0:
            print("The weight file {} is loaded.".format(config["MODEL_PATH"]))

    ## Initialize dataloaders specific to either whole-slide training or MIL.
    if is_rank0:
        print("### Initializing dataloaders ###")
    if config["USE_MIL"]:
        train_dataloader = MILDataloader(
            dataset=train_dataset,
            augment=True,
            shuffle=True,
            num_classes=config["NUM_CLASSES"],
            batch_size=config["BATCH_SIZE"],
            snapshot_path=(config["DEBUG_PATH"] if is_rank0 else None),
            hvd=hvd,
            mil_model=model,
            mil_patch_size=config["MIL_PATCH_SIZE"],
            mil_infer_batch_size=config["MIL_INFER_BATCH_SIZE"],
            mil_use_em=config["MIL_USE_EM"],
            mil_k=config["MIL_K"],
            mil_skip_white=config["MIL_SKIP_WHITE"],
        )
        val_dataloader = MILDataloader(
            dataset=val_dataset,
            augment=False,
            shuffle=False,
            num_classes=config["NUM_CLASSES"],
            batch_size=config["BATCH_SIZE"],
            snapshot_path=(config["DEBUG_PATH"] if is_rank0 else None),
            hvd=hvd,
            mil_model=model,
            mil_patch_size=config["MIL_PATCH_SIZE"],
            mil_infer_batch_size=config["MIL_INFER_BATCH_SIZE"],
            mil_k=1, # K=1 for validation
        )
    else:
        train_dataloader = WholeSlideDataloader(
            dataset=train_dataset,
            augment=True,
            shuffle=True,
            num_classes=config["NUM_CLASSES"],
            batch_size=config["BATCH_SIZE"],
            snapshot_path=(config["DEBUG_PATH"] if is_rank0 else None),
            hvd=hvd,
        )
        val_dataloader = WholeSlideDataloader(
            dataset=val_dataset,
            augment=False,
            shuffle=False,
            num_classes=config["NUM_CLASSES"],
            batch_size=config["BATCH_SIZE"],
            snapshot_path=(config["DEBUG_PATH"] if is_rank0 else None),
            hvd=hvd,
        )
    print(
        "Train dataloader contains {} slides on rank {}. The first one is {}.".format(
            len(train_dataloader),
            hvd.rank() if hvd != None else 0,
            train_dataloader.get_slide_path(0),
        )
    )
    print(
        "Val dataloader contains {} slides on rank {}. The first one is {}.".format(
            len(val_dataloader),
            hvd.rank() if hvd != None else 0,
            val_dataloader.get_slide_path(0),
        )
    )

    ## Start training
    if is_rank0:
        print("### Start training ###")

    optimizer = tf.keras.optimizers.Adam(learning_rate=config["INIT_LEARNING_RATE"])
    if hvd != None:
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            device_dense=("/cpu:0" if config["USE_HMS"] else ""),
            device_sparse=("/cpu:0" if config["USE_HMS"] else ""),
        )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    callbacks = []
    if config["USE_HMS"]:
        callbacks.append(HMSTFKerasCallback(hvd=hvd, default_batch_size=config["BATCH_SIZE"]))
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
            filepath=config["MODEL_PATH"],
            factor=config["REDUCE_LR_FACTOR"],
            min_lr=1e-8, 
            monitor="val_loss",
            patience=config["REDUCE_LR_PATIENCE"],
            verbose=(1 if is_rank0 else 0),
        )
    )
    if is_rank0:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=config["MODEL_PATH"],
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=(1 if is_rank0 else 0),
            )
        )
        callbacks.append(
            TimeRecorder(
                filepath=config["TIME_RECORD_PATH"],
                train_phase=True,
                skip_first_n=10,
                include_load=config["USE_MIL"],
            )
        )
        callbacks.append(
            TimeRecorder(
                filepath=config["TEST_TIME_RECORD_PATH"],
                train_phase=False,
                skip_first_n=10,
                include_load=config["USE_MIL"],
            )
        )

    if config["USE_MIL"]:
        model.fit(
            train_dataloader,
            workers=0, # MIL dataloader should be in the main thread
            max_queue_size=1,
            use_multiprocessing=False,
            epochs=config["EPOCHS"],
            steps_per_epoch=config["NUM_UPDATES_PER_EPOCH"],
            validation_data=val_dataloader,
            callbacks=callbacks,
            shuffle=False, # Shuffling is already done in dataloader
            verbose=(1 if is_rank0 else 0),
        )
    else:
        model.fit(
            train_dataloader,
            workers=1,
            max_queue_size=1,
            use_multiprocessing=False,
            epochs=config["EPOCHS"],
            steps_per_epoch=config["NUM_UPDATES_PER_EPOCH"],
            validation_data=val_dataloader,
            callbacks=callbacks,
            shuffle=False, # Shuffling is already done in dataloader
            verbose=(1 if is_rank0 else 0),
        )
