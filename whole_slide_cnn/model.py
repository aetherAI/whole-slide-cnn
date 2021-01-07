import numpy as np
import tensorflow as tf

from whole_slide_cnn.resnet import (
    ResNet34,
    ResNet50,
)

graph_mapping = {
    "resnet34": lambda *args, **kwargs: ResNet34(
        *args, 
        norm_use="bn", 
        weights=None,
        use_fixup=False,
        data_format="channels_last",
        **kwargs
    ),
    "fixup_resnet34": lambda *args, **kwargs: ResNet34(
        *args, 
        norm_use="", 
        weights=None,
        use_fixup=True,
        data_format="channels_last",
        **kwargs
    ),
    "fixup_resnet50": lambda *args, **kwargs: ResNet50(
        *args, 
        norm_use="", 
        weights="imagenet",
        use_fixup=True, 
        data_format="channels_last",
        **kwargs
    ),
}


class MaxFeatBlockDescriptorLayer(tf.keras.layers.Layer):
    def __init__(self, emb_shape, n_classes, tau=0.3, **kwargs):
        super(MaxFeatBlockDescriptorLayer, self).__init__(**kwargs)
        self.emb_shape = emb_shape
        self.n_classes = n_classes
        self.tau = tau

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        emb, prob_map = inputs
        emb_flat = tf.reshape(emb, [-1, self.emb_shape[0] * self.emb_shape[1], self.emb_shape[2]])
        prob_map_flat = tf.reshape(prob_map, [-1, self.emb_shape[0] * self.emb_shape[1], self.n_classes])

        argmax_prob_map_flat = tf.argmax(prob_map_flat, axis=1) # shape: [batch_size, n_classes]
        maxfeat = tf.map_fn(
            fn=lambda tensors: tf.gather(
                tensors[0],
                tensors[1],
                axis=0,
            ),
            elems=[emb_flat, argmax_prob_map_flat],
            dtype=tf.float32,
        ) # shape: [batch_size, n_classes, emb_dim]
        print(maxfeat.shape)

        avg_prob = tf.reduce_mean(prob_map_flat, axis=1) # shape: [batch_size, n_classes]
        is_representitive = avg_prob > self.tau # shape: [batch_size, n_classes]

        maxfeat = tf.cast(is_representitive[:, :, tf.newaxis], tf.float32) * maxfeat

        return maxfeat


def build_model(
    input_shape,
    n_classes,
    backbone="resnet34",
    pool_use="gmp",
    use_mixed_precision=False,
    batch_size=None,
    use_huge_layer_wrapper=False,
    for_post_train=False,
):
    model_fn = graph_mapping[backbone]

    def get_conv_block(image_shape):
        conv_block = model_fn(
            input_shape=image_shape,
            use_mixed_precision=use_mixed_precision,
            use_huge_layer_wrapper=use_huge_layer_wrapper,
        )
        return conv_block

    def get_post_conv_block(emb_shape):
        emb = tf.keras.Input(shape=emb_shape)

        # Since the number of non-cancer case is more than the other two,
        # we set a higher bias on non-cancer at model initialization to 
        # speed up convergence.
        initial_bias_value = np.zeros([n_classes])
        initial_bias_value[0] = 1.0

        if pool_use == "gmp":
            pool = tf.keras.layers.GlobalMaxPooling2D(name='GMP_layer')(emb)
            out_layer = tf.keras.layers.Dense(
                units=n_classes, 
                activation='softmax', 
                name='output',
                bias_initializer=tf.keras.initializers.Constant(initial_bias_value),
            )
            out = out_layer(pool)
        elif pool_use == "gap":
            pool = tf.keras.layers.GlobalAveragePooling2D(name='GAP_layer')(emb)
            out_layer = tf.keras.layers.Dense(
                units=n_classes, 
                activation='softmax', 
                name='output',
                bias_initializer=tf.keras.initializers.Constant(initial_bias_value),
            )
            out = out_layer(pool)
        else:
            raise NotImplementedError("{} is not a supported pooling layer.".format(pool_use))

        if for_post_train:
            prob_map = out_layer(emb)
            maxfeat = MaxFeatBlockDescriptorLayer(emb_shape, n_classes)([emb, prob_map])
            return tf.keras.Model(inputs=emb, outputs=[out, pool, maxfeat])
        else:
            return tf.keras.Model(inputs=emb, outputs=out)

    conv_block = get_conv_block(input_shape)
    post_conv_block = get_post_conv_block(conv_block.output.shape[1: ])
    output = post_conv_block(conv_block.output)
    model = tf.keras.Model(inputs=conv_block.input, outputs=output)

    return model
