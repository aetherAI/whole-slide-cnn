import cv2
from imgaug import augmenters as iaa
from mpi4py import MPI
import numpy as np
import os
from PIL import Image, ImageOps
import tensorflow as tf
import time
import traceback

def preprocess_input(img):
    return (np.array(img).astype(np.float32) - 127.5) / 127.5

def inverse_preprocess_input(img):
    x = np.array(img) * 127.5 + 127.5
    x = np.minimum(255.0, np.maximum(0.0, x))
    x = x.astype(np.uint8)
    return x

class WholeSlideDataloader(tf.keras.utils.Sequence):
    def __init__(
        self, 
        dataset,
        augment,
        shuffle,
        num_classes,
        batch_size=1,
        snapshot_path=None,
        hvd=None,
    ):
        self.dataset = dataset
        self.augment = augment
        self.shuffle = shuffle
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.snapshot_path = snapshot_path
        self.hvd = hvd

        self._reinit_shuffle_list()

    def __len__(self):
        num_workers = self.hvd.size() if self.hvd != None else 1
        return len(self.dataset) // self.batch_size // num_workers

    def __getitem__(self, idx):
        begin_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size

        x_batch = []
        y_batch = []
        for shuffle_list_idx in range(begin_idx, end_idx):
            dataset_idx = self.shuffle_list[shuffle_list_idx]
            loaded = False
            while not loaded:
                try:
                    img, label = self.dataset[dataset_idx]
                    loaded = True
                except Exception as e:
                    print(traceback.format_exc())
                    print(
                        "Error occurs while loading {} . Retry after 5 seconds.".format(
                            self.dataset.get_slide_path(dataset_idx)
                        )
                    )
                    time.sleep(5)

            if self.augment:
                if img.size < 4 * 1024 * 1024 * 1024:
                    img = _get_augmentor().augment_image(img)
                else:
                    # If the image is too large, imgaug will fail on affine transformation. Use PIL instead.
                    img = _get_augmentor_wo_affine().augment_image(img)
                    img = Image.fromarray(img)
                    angle = np.random.uniform(0.0, 360.0)
                    translate = tuple(np.random.randint(-220, 220, size=[2]))
                    img = img.rotate(angle, resample=Image.NEAREST, translate=translate, fillcolor=(255, 255, 255))
                    img = np.array(img)

            if self.snapshot_path != None:
                os.makedirs(self.snapshot_path, exist_ok=True)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    os.path.join(self.snapshot_path, "dataloader_snapshot.tiff"),
                    img_bgr,
                )

            img = preprocess_input(img)
            y = np.zeros(shape=[self.num_classes])
            y[label] = 1.0
            
            x_batch.append(img)
            y_batch.append(y)

        return np.array(x_batch), np.array(y_batch)

    def on_epoch_end(self):
        self._reinit_shuffle_list()

    def get_slide_path(self, idx):
        dataset_idx = self.shuffle_list[idx]
        return self.dataset.get_slide_path(dataset_idx)

    def get_y_true(self, idx):
        dataset_idx = self.shuffle_list[idx]
        return self.dataset.get_y_true(dataset_idx)

    def get_dataset_idx(self, idx):
        return self.shuffle_list[idx]

    def _reinit_shuffle_list(self):
        self.shuffle_list = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.shuffle_list)
        if self.hvd != None:
            self.shuffle_list = MPI.COMM_WORLD.bcast(self.shuffle_list, root=0)

            num_workers = self.hvd.size()
            rank = self.hvd.rank()
            self.shuffle_list = [
                self.shuffle_list[idx * num_workers + rank]
                for idx in range(len(self))
            ]


class MILDataloader(WholeSlideDataloader):
    MIL_WHITE_THRESHOLD = 220
    MIL_EM_GAUSSIAN_KERNEL_SIZE = 3
    MIL_EM_P1 = 0.1
    MIL_EM_P2 = 0.05

    def __init__(
        self, 
        dataset,
        augment,
        shuffle,
        num_classes,
        mil_model,
        batch_size=1,
        snapshot_path=None,
        hvd=None,
        mil_patch_size=[224, 224],
        mil_infer_batch_size=32,
        mil_use_em=False,
        mil_k=1,
        mil_skip_white=True,
    ):
        super(MILDataloader, self).__init__(
            dataset=dataset,
            augment=augment,
            shuffle=shuffle,
            num_classes=num_classes,
            batch_size=batch_size,
            snapshot_path=snapshot_path,
            hvd=hvd,
        )
        self.mil_model = mil_model
        self.mil_patch_size = mil_patch_size
        self.mil_infer_batch_size = mil_infer_batch_size
        self.mil_use_em = mil_use_em
        self.mil_k = mil_k
        self.mil_skip_white = mil_skip_white
            
    def __getitem__(self, idx):
        x_batch_wsi, y_batch_wsi = super(MILDataloader, self).__getitem__(idx)

        x_batch = []
        y_batch = []
        for i in range(self.batch_size):
            x = x_batch_wsi[i]
            y = y_batch_wsi[i]

            num_patch_y = x.shape[0] // self.mil_patch_size[1]
            num_patch_x = x.shape[1] // self.mil_patch_size[0]

            mil_infer_dataset = []
            coords = []
            for i in range(num_patch_y):
                for j in range(num_patch_x):
                    sliced_x = x[
                        i * self.mil_patch_size[1]: (i + 1) * self.mil_patch_size[1],
                        j * self.mil_patch_size[0]: (j + 1) * self.mil_patch_size[0],
                        :,
                    ]
                    if (
                        self.mil_skip_white and 
                        np.min(sliced_x) > preprocess_input(self.MIL_WHITE_THRESHOLD)
                    ):
                        continue
                    mil_infer_dataset.append(sliced_x)
                    coords.append((j, i))
            mil_infer_dataset = np.array(mil_infer_dataset)

            mil_infer_res = []
            for begin_idx in range(0, len(mil_infer_dataset), self.mil_infer_batch_size):
                end_idx = np.minimum(len(mil_infer_dataset), begin_idx + self.mil_infer_batch_size)
                mil_infer_res.append(
                    self.mil_model.predict_on_batch(
                         mil_infer_dataset[begin_idx: end_idx]
                    )
                )
            mil_infer_res = np.concatenate(mil_infer_res, axis=0)
            benign_rate = mil_infer_res[:, 0]

            if not self.mil_use_em:
                top_k_indices = np.argsort(benign_rate)[0: self.mil_k]
                for index in top_k_indices:
                    x_batch.append(mil_infer_dataset[index])
                    y_batch.append(y)
            else:
                res_map = np.zeros([num_patch_y, num_patch_x, self.num_classes - 1]) # Excluding non-cancer
                for i in range(len(mil_infer_res)):
                    res = mil_infer_res[i]
                    coord = coords[i]
                    res_map[coord[1], coord[0], :] = res[1: ]

                res_map_blurred = cv2.GaussianBlur(
                    res_map, 
                    (self.MIL_EM_GAUSSIAN_KERNEL_SIZE, self.MIL_EM_GAUSSIAN_KERNEL_SIZE),
                    0
                )

                mil_infer_res_blurred = []
                for i in range(len(mil_infer_res)):
                    coord = coords[i]
                    res_blurred = res_map_blurred[coord[1], coord[0], :]
                    mil_infer_res_blurred.append(res_blurred)
                mil_infer_res_blurred = np.array(mil_infer_res_blurred)

                thres_p1 = np.percentile(mil_infer_res_blurred, 100.0 - self.MIL_EM_P1)

                select = None
                for class_id in range(1, self.num_classes):
                    candidates = mil_infer_res_blurred[:, class_id - 1].tolist()
                    if self.hvd != None:
                        candidates = MPI.COMM_WORLD.gather(candidates, root=0)
                        candidates = MPI.COMM_WORLD.bcast(candidates, root=0)
                        flatten = []
                        for candidate in candidates:
                            flatten += candidate
                        candidates = flatten
                    candidates = np.array(candidates)

                    thres_p2 = np.percentile(candidates, 100.0 - self.MIL_EM_P2)
                    thres = np.minimum(thres_p1, thres_p2)

                    if select is None:
                        select = mil_infer_res_blurred[:, class_id - 1] > thres
                    else:
                        select = np.logical_or(select, mil_infer_res_blurred[:, class_id - 1] > thres)

                no_selected = True
                for i in range(len(mil_infer_res_blurred)):
                    is_select = select[i]
                    if is_select:
                        x_batch.append(mil_infer_dataset[i])
                        y_batch.append(y)
                        no_selected = False

                if no_selected:
                    x_batch.append(mil_infer_dataset[np.argmin(benign_rate)])
                    y_batch.append(y)

        x_batch = np.array(x_batch) # The batch dimension is large as (self.batch_size * self.mil_k).
        y_batch = np.array(y_batch)

        if self.snapshot_path != None:
            os.makedirs(self.snapshot_path, exist_ok=True)
            img = inverse_preprocess_input(x_batch[0])
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                os.path.join(self.snapshot_path, "mil_top_patch_snapshot.tiff"),
                img_bgr,
            )

        return x_batch, y_batch


class MILVerboseDataloader(WholeSlideDataloader):
    MIL_WHITE_THRESHOLD = 220

    def __init__(
        self, 
        dataset,
        augment,
        shuffle,
        num_classes,
        mil_model,
        batch_size=1,
        snapshot_path=None,
        hvd=None,
        mil_patch_size=[224, 224],
        mil_infer_batch_size=32,
        mil_skip_white=True,
    ):
        super(MILVerboseDataloader, self).__init__(
            dataset=dataset,
            augment=augment,
            shuffle=shuffle,
            num_classes=num_classes,
            batch_size=batch_size,
            snapshot_path=snapshot_path,
            hvd=hvd,
        )
        self.mil_model = mil_model
        self.mil_patch_size = mil_patch_size
        self.mil_infer_batch_size = mil_infer_batch_size
        self.mil_skip_white = mil_skip_white

    def __getitem__(self, idx):
        x_batch_wsi, y_batch_wsi = super(MILVerboseDataloader, self).__getitem__(idx)

        output_map_batch = [] # shape: [batch_size, num_tile_y, num_tile_x, num_classes]
        emb_map_batch = [] # shape: [batch_size, num_tile_y, num_tile_x, 4096]
        maxfeat_map_batch = [] # shape: [batch_size, num_tile_y, num_tile_x, num_classes, 4096]
        y_batch = [] # shape: [batch_size, num_classes]
        for i in range(self.batch_size):
            x = x_batch_wsi[i]
            y = y_batch_wsi[i]

            num_patch_y = x.shape[0] // self.mil_patch_size[1]
            num_patch_x = x.shape[1] // self.mil_patch_size[0]

            mil_infer_dataset = []
            coords = []
            for i in range(num_patch_y):
                for j in range(num_patch_x):
                    sliced_x = x[
                        i * self.mil_patch_size[1]: (i + 1) * self.mil_patch_size[1],
                        j * self.mil_patch_size[0]: (j + 1) * self.mil_patch_size[0],
                        :,
                    ]
                    if (
                        self.mil_skip_white and 
                        np.min(sliced_x) > preprocess_input(self.MIL_WHITE_THRESHOLD)
                    ):
                        continue
                    mil_infer_dataset.append(sliced_x)
                    coords.append((j, i))
            mil_infer_dataset = np.array(mil_infer_dataset)

            mil_infer_res = []
            mil_infer_emb = []
            mil_infer_maxfeat = []
            for begin_idx in range(0, len(mil_infer_dataset), self.mil_infer_batch_size):
                end_idx = np.minimum(len(mil_infer_dataset), begin_idx + self.mil_infer_batch_size)
                res, emb, maxfeat = self.mil_model.predict_on_batch(
                    mil_infer_dataset[begin_idx: end_idx]
                )
                mil_infer_res.append(res)
                mil_infer_emb.append(emb)
                mil_infer_maxfeat.append(maxfeat)

            mil_infer_res = np.concatenate(mil_infer_res, axis=0)
            mil_infer_emb = np.concatenate(mil_infer_emb, axis=0)
            mil_infer_maxfeat = np.concatenate(mil_infer_maxfeat, axis=0)

            output_map = np.concatenate(
                [
                    np.ones([num_patch_y, num_patch_x, 1]),
                    np.zeros([num_patch_y, num_patch_x, self.num_classes - 1]),
                ],
                axis=-1
            )
            emb_dim = self.mil_model.outputs[1].shape[-1]
            emb_map = np.zeros([num_patch_y, num_patch_x, emb_dim])
            maxfeat_map = np.zeros([num_patch_y, num_patch_x, self.num_classes, emb_dim])

            for i in range(len(mil_infer_res)):
                coord = coords[i]
                output_map[coord[1], coord[0], :] = mil_infer_res[i]
                emb_map[coord[1], coord[0], :] = mil_infer_emb[i]
                maxfeat_map[coord[1], coord[0], :] = mil_infer_maxfeat[i]

            output_map_batch.append(output_map)
            emb_map_batch.append(emb_map)
            maxfeat_map_batch.append(maxfeat_map)
            y_batch.append(y)
    
        output_map_batch = np.array(output_map_batch)
        emb_map_batch = np.array(emb_map_batch)
        maxfeat_map_batch = np.array(maxfeat_map_batch)
        y_batch = np.array(y_batch)

        return (output_map_batch, emb_map_batch, maxfeat_map_batch), y_batch


def _get_augmentor():
    augmentation = iaa.Sequential([
        iaa.Fliplr(0.5, name="FlipLR"),
        iaa.Flipud(0.5, name="FlipUD"),
        iaa.Affine(
            translate_percent=(-0.025, 0.025),
            rotate=(0, 360),
            order=3, # Bicubic interpolation
            cval=255,
            mode='constant',
            name='affine',
        ),
        iaa.contrast.LinearContrast(alpha=(0.5, 1.5)),
        iaa.color.MultiplyBrightness(mul=(0.65, 1.35)),
        iaa.color.AddToHueAndSaturation(value=(-32, 32), per_channel=True),
    ])    
    return augmentation

def _get_augmentor_wo_affine():
    augmentation = iaa.Sequential([
        iaa.Fliplr(0.5, name="FlipLR"),
        iaa.Flipud(0.5, name="FlipUD"),
        iaa.contrast.LinearContrast(alpha=(0.5, 1.5)),
        iaa.color.MultiplyBrightness(mul=(0.65, 1.35)),
        iaa.color.AddToHueAndSaturation(value=(-32, 32), per_channel=True),
    ])    
    return augmentation
