import cv2
import numpy as np
import tensorflow as tf
import time

class ReduceLRAndRollbackOnPlateau(tf.keras.callbacks.ReduceLROnPlateau):
    def __init__(self, filepath, **kwargs):
        super(ReduceLRAndRollbackOnPlateau, self).__init__(**kwargs)
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        old_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        super(ReduceLRAndRollbackOnPlateau, self).on_epoch_end(epoch, logs)
        cur_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        if cur_lr < old_lr:
            print('\nThe weight is rollbacked.\n')
            self.model.load_weights(self.filepath)

class TimeRecorder(tf.keras.callbacks.Callback):
    def __init__(self, filepath, train_phase=True, skip_first_n=0, include_load=False, **kwargs):
        super(TimeRecorder, self).__init__(**kwargs)
        self.filepath = filepath
        self.train_phase = train_phase
        self.skip_first_n = skip_first_n
        self.include_load = include_load

        self.file = None
        self.batch_idx = 0
        self.begin_time = None
        self.durations = []

        if train_phase:
            self.on_train_begin = self._on_begin
            self.on_train_batch_begin = self._on_batch_begin
            self.on_train_batch_end = self._on_batch_end
            self.on_train_end = self._on_end
        else:
            self.on_test_begin = self._on_begin
            self.on_test_batch_begin = self._on_batch_begin
            self.on_test_batch_end = self._on_batch_end
            self.on_test_end = self._on_end

    def _on_begin(self, logs=None):
        if self.filepath != None:
            self.file = open(self.filepath, "w")
            self.file.write("batch_idx,duration\n")
            self.file.flush()

        if self.include_load:
            self.begin_time = time.time()

    def _on_batch_begin(self, batch, logs=None):
        if not self.include_load:
            self.begin_time = time.time()

    def _on_batch_end(self, batch, logs=None):
        end_time = time.time()
        duration = end_time - self.begin_time
        if self.batch_idx >= self.skip_first_n:
            self.durations.append(duration)
        if self.filepath != None:
            self.file.write("{},{}\n".format(self.batch_idx, duration))
            self.file.flush()
        self.batch_idx += 1

        if self.include_load:
            self.begin_time = time.time()

    def _on_end(self, logs=None):
        if self.filepath != None:
            self.file.close()

        self.file = None
        self.batch_idx = 0

        throughputs = 60.0 / np.array(self.durations)
        mean = np.mean(throughputs)
        stddev = np.std(throughputs, ddof=1)
        ci_low = mean - 1.96 * stddev
        ci_high = mean + 1.96 * stddev
        print("{} Throughput: {} ({} - {}) images/minutes.".format(
            "Train" if self.train_phase else "Test",
            mean, 
            ci_low, 
            ci_high
        ))

def get_cam(
    input_image,
    model,
    class_select=0,
):
    emb_layer = None
    for layer in model.layers:
        if len(layer.output.shape) == 4:
            emb_layer = layer
    get_emb_function = tf.keras.backend.function([model.input], emb_layer.output)
    emb = get_emb_function([input_image[np.newaxis, :, :, :]])[0]

    emb_input = tf.keras.Input(shape=emb.shape)
    final_dense_layer = model.layers[-1].layers[-1]
    cam = final_dense_layer(emb_input)

    cam_model = tf.keras.Model(inputs=emb_input, outputs=cam)
    cam = cam_model.predict_on_batch(emb[np.newaxis, :, :, :])[0]

    cam_selected = cam[:, :, class_select]
    return cam_selected

def get_mil_pred_map(
    input_image,
    model,
    class_select=0,
):
    y_size, x_size, _ = input_image.shape
    _, y_patch_size, x_patch_size, _ = model.input.shape

    pred_map = np.zeros(shape=(y_size // y_patch_size, x_size // x_patch_size))
    for i in range(y_size // y_patch_size):
        for j in range(x_size // x_patch_size):
            patch = input_image[
                i * y_patch_size: (i + 1) * y_patch_size,
                j * x_patch_size: (j + 1) * x_patch_size,
                :
            ]
            patch_pred = model.predict_on_batch(patch[np.newaxis, :, :, :])[0]
            selected_patch_pred = patch_pred[class_select]
            pred_map[i, j] = selected_patch_pred

    return pred_map

def draw_pred_map(
    pred_map,
    size=None,
    threshold=0.5,
):
    if size != None:
        pred_map = cv2.resize(pred_map, tuple(size), interpolation=cv2.INTER_CUBIC)
    pred_map[pred_map < threshold] = 0.0

    pred_map = np.maximum(0.0, np.minimum(255.0, (pred_map * 255))).astype(np.uint8)
    colormap_bgr = cv2.applyColorMap(pred_map, cv2.COLORMAP_JET)
    alphamap = pred_map[:, :, np.newaxis]

    heatmap_bgra = np.concatenate([colormap_bgr, alphamap], axis=-1)
    return heatmap_bgra
