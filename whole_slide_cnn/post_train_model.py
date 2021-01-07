from mpi4py import MPI
import numpy as np
import tensorflow as tf
import tqdm

def build_milrnn_model(
    input_shape,
    emb_dim,
    n_classes,
    top_k=10,
    hidden_count=128,
):
    pred_shape = input_shape + [n_classes]
    emb_shape = input_shape + [emb_dim]

    pred_input = tf.keras.Input(shape=pred_shape)
    emb_input = tf.keras.Input(shape=emb_shape)

    spatial_size = input_shape[0] * input_shape[1]

    malignant_map = 1.0 - pred_input[:, :, :, 0]
    malignant_map_flat = tf.reshape(malignant_map, [-1, spatial_size])
    _, top_k_indices = tf.math.top_k(
        malignant_map_flat, 
        k=top_k,
        sorted=True,
    )

    emb_input_flat = tf.reshape(emb_input, [-1, spatial_size, emb_dim])
    top_k_emb = tf.gather(emb_input_flat, top_k_indices, batch_dims=1, axis=1)
    top_k_emb = tf.unstack(top_k_emb, axis=1)

    state = None
    for i in range(top_k):
        x = tf.keras.layers.Dense(
            hidden_count, 
            name="input_dense_{}".format(i)
        )(top_k_emb[i])
        if state != None:
            state_fc = tf.keras.layers.Dense(
                hidden_count, 
                name="state_dense_{}".format(i)
            )(state)
            x += state_fc
        state = tf.nn.relu(x)
    output = tf.keras.layers.Dense(
        n_classes, 
        activation="softmax", 
        name="output_dense_{}".format(i)
    )(state)

    model = tf.keras.Model(inputs=[pred_input, emb_input], outputs=output)

    return model

def to_histogram(dataloader, is_rank0):
    histogram_list = []
    pred_list = []
    for (output_map_batch, _, _), y_batch in tqdm.tqdm(dataloader, disable=(not is_rank0)):
        histogram_batch = np.sum(np.sum(output_map_batch, axis=1), axis=1)

        pred_batch = np.argmax(y_batch, axis=-1)

        histogram_list.append(histogram_batch)
        pred_list.append(pred_batch)

    histogram_list = np.concatenate(histogram_list, axis=0)
    pred_list = np.concatenate(pred_list, axis=0)

    return histogram_list, pred_list

def to_maxfeat_feature(dataloader, is_rank0):
    feature_list = []
    pred_list = []
    for (_, _, maxfeat_batch), y_batch in tqdm.tqdm(dataloader, disable=(not is_rank0)):
        # shape(maxfeat_batch): [batch_size, n_tiles_y, n_tiles_x, n_classes, emb_dim]
        is_valid = (np.sum(maxfeat_batch, axis=-1) != 0) # shape: [batch_size, n_tiles_y, n_tiles_x, n_classes]
        num_valid = np.sum(np.sum(is_valid.astype(np.float32), axis=1), axis=1) # shape: [batch_size, n_classes]
        num_valid = np.maximum(num_valid, 1.0)

        class_descriptor = (
            np.sum(np.sum(maxfeat_batch, axis=1), axis=1) / 
            num_valid[:, :, np.newaxis]
        ) # shape: [batch_size, n_classes, emb_dim]
        feature = np.reshape(
            class_descriptor, 
            [
                class_descriptor.shape[0], 
                class_descriptor.shape[1] * class_descriptor.shape[2],
            ]
        ) # shape: [batch_size, n_classes * emb_dim]

        pred_batch = np.argmax(y_batch, axis=-1)

        feature_list.append(feature)
        pred_list.append(pred_batch)

    feature_list = np.concatenate(feature_list, axis=0)
    pred_list = np.concatenate(pred_list, axis=0)

    return feature_list, pred_list
