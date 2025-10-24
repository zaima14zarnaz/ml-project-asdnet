import tensorflow as tf
from tensorflow import keras
from tensorflow import nest 
from tensorflow.keras import layers
import os
import re
import datetime
import multiprocessing

import tensorflow as tf
from tensorflow import keras

class _AddScalarLoss(keras.layers.Layer):
    def __init__(self, weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.weight = float(weight)
    def call(self, x, training=None):
        xs = tf.nest.flatten(x)
        pieces = [tf.reduce_mean(tf.cast(t, tf.float32)) for t in xs]
        self.add_loss(tf.add_n(pieces) * tf.cast(self.weight, tf.float32))
        return x  # passthrough
    def get_config(self):
        cfg = super().get_config(); cfg.update({"weight": self.weight}); return cfg

class _AddL2Tap(keras.layers.Layer):
    def __init__(self, weights_to_reg, weight_decay, **kwargs):
        super().__init__(**kwargs)
        # don't let Keras track this plain Python list
        self.weights_to_reg = self._no_dependency(list(weights_to_reg))
        self.weight_decay = float(weight_decay)
    def call(self, x, training=None):
        regs = []
        for w in self.weights_to_reg:
            n = w.name or ""
            if "gamma" in n or "beta" in n:  # skip BN gamma/beta
                continue
            regs.append(keras.regularizers.l2(self.weight_decay)(w) / tf.cast(tf.size(w), tf.float32))
        if regs:
            self.add_loss(tf.add_n(regs))
        return x  # passthrough
    def get_config(self):
        cfg = super().get_config(); cfg.update({"weight_decay": self.weight_decay}); return cfg

class _PassThrough(keras.layers.Layer):
    def call(self, inputs, training=None):
        # inputs = [main_output, side_dep]
        return inputs[0]

class LossHook(layers.Layer):
    def __init__(self, weight=1.0, **kw):
        super().__init__(**kw)
        self.weight = float(weight)

    def call(self, inputs):
        t, anchor = inputs  # t: loss source; anchor: any tensor already on the path to outputs
        parts = [tf.cast(p, tf.float32) for p in tf.nest.flatten(t)]
        loss_tensor = tf.add_n([tf.reduce_mean(p) for p in parts]) * self.weight
        self.add_loss(loss_tensor)
        # Optional: log metric
        self.add_metric(tf.add_n([tf.reduce_mean(p) for p in parts]),
                        name=self.name + "_mean", aggregation='mean')
        return anchor  # passthrough so this layer stays in the graph


def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}  {}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else "",
            array.dtype))
    print(text)


class ASRNet():
    def __init__(self, mode, config, model_dir, keras_model, model_name):
        self.mode = mode
        self.config = config
        self.model_dir = model_dir

        # Model is pre-built and passed from "train.py" or its variants
        self.keras_model = keras_model

        self.model_name = model_name

        self.set_log_dir()

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/[\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/" + self.model_name + "_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(self.config.NAME.lower(), now))

        # Create log_dir if not exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "{}_{}_*epoch*.h5".format(self.model_name,
                                                                                    self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")

    def train(self, train_generator, val_generator, learning_rate, epochs, layers, custom_callbacks=None):

        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        logs_path = self.log_dir + "/training.log"

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=1, save_weights_only=True),
            keras.callbacks.CSVLogger(logs_path, separator=",", append=True),
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        num = self.num_model_outputs
        def wrap_with_dummies(gen, num_outputs):
            # gen yields either (x, y) or just x; we ignore y and add a tuple of dummy targets
            for i, batch in enumerate(gen):
                if isinstance(batch, tuple) and len(batch) == 2:
                    # print(f'both outputs {i}')
                    x, _ = batch
                else:
                    x = batch
                # Use scalar float32 dummies; shapes don't matter since loss=None
                dummies = tuple(tf.constant(0.0, dtype=tf.float32) for _ in range(num_outputs))
                yield x, dummies

        train_gen = wrap_with_dummies(train_generator, num)
        val_gen   = wrap_with_dummies(val_generator,   num)

        workers = 1
        self.keras_model.fit_generator(
            train_gen,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_gen,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=False,
        )
        self.epoch = max(self.epoch, epochs)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    # def compile(self, learning_rate, momentum):
    #     """Gets the model ready for training. Adds losses, regularization, and
    #     metrics. Then calls the Keras compile() function.
    #     """

    #     # Optimizer object
    #     optimizer = keras.optimizers.SGD(
    #         lr=learning_rate, momentum=momentum,
    #         clipnorm=self.config.GRADIENT_CLIP_NORM)

    #     # Add Losses
    #     # First, clear previously set losses to avoid duplication
    #     self.keras_model._losses = []
    #     self.keras_model._per_input_losses = {}

    #     loss_names = ["rank_loss"]

    #     # for name in loss_names:
    #     #     layer = self.keras_model.get_layer(name)
    #     #     if layer.output in self.keras_model.losses:
    #     #         continue
    #     #     loss = (
    #     #             tf.reduce_mean(layer.output, keepdims=True)
    #     #             * self.config.LOSS_WEIGHTS.get(name, 1.))
    #     #     self.keras_model.add_loss(loss)
    #     for name in loss_names:
    #         self.add_layer_output_loss(
    #             self.keras_model,
    #             name,
    #             self.config.LOSS_WEIGHTS.get(name, 1.0)
    #         )

    #     # Add L2 Regularization
    #     # Skip gamma and beta weights of batch normalization layers.
    #     reg_losses = [
    #         keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
    #         for w in self.keras_model.trainable_weights
    #         if 'gamma' not in w.name and 'beta' not in w.name]
    #     self.keras_model.add_loss(tf.add_n(reg_losses))

    #     # Compile
    #     self.keras_model.compile(
    #         optimizer=optimizer,
    #         loss=[None] * len(self.keras_model.outputs))

    #     # Add metrics for losses
    #     for name in loss_names:
    #         if name in self.keras_model.metrics_names:
    #             continue
    #         layer = self.keras_model.get_layer(name)
    #         self.keras_model.metrics_names.append(name)
    #         loss = (
    #                 tf.reduce_mean(layer.output, keepdims=True)
    #                 * self.config.LOSS_WEIGHTS.get(name, 1.))
    #         self.keras_model.metrics_tensors.append(loss)

    def _attach_losses_into_graph(self):
        if self._losses_attached:
            return
        # avoid double-attach if method called twice
        if any(l.name == "rank_loss_tap" for l in self.keras_model.layers):
            self._losses_attached = True
            return

        # 1) tap the source that used to feed your custom loss
        rank_src = self.keras_model.get_layer("rank_loss").output

        side = _AddScalarLoss(
            weight=self.config.LOSS_WEIGHTS.get("rank_loss", 1.0),
            name="rank_loss_tap"
        )(rank_src)

        # 2) add L2 via tap (optional if you already use kernel_regularizer)
        side = _AddL2Tap(
            weights_to_reg=self.keras_model.trainable_weights,
            weight_decay=self.config.WEIGHT_DECAY,
            name="l2_reg_tap"
        )(side)

        # 3) keep outputs identical but depend on side path
        new_outputs = [
            _PassThrough(name=f"passthrough_{i}")([o, side])
            for i, o in enumerate(self.keras_model.outputs)
        ]
        self.keras_model = keras.Model(self.keras_model.inputs, new_outputs, name=self.keras_model.name)
        self._losses_attached = True

    # def compile(self, learning_rate, momentum):
    #     optimizer = keras.optimizers.SGD(
    #         lr=learning_rate, momentum=momentum,
    #         clipnorm=self.config.GRADIENT_CLIP_NORM)

    #     self.keras_model._losses = []
    #     self.keras_model._per_input_losses = {}

    #     loss_names = ["rank_loss"]

    #     def scalar_from_layer_output(model, layer_name, weight):
    #         layer = model.get_layer(layer_name)
    #         outs = tf.nest.flatten(layer.output)
    #         pieces = [tf.reduce_mean(tf.cast(t, tf.float32)) for t in outs]
    #         return tf.add_n(pieces) * tf.cast(weight, tf.float32)

    #     for name in loss_names:
    #         w = self.config.LOSS_WEIGHTS.get(name, 1.0)
    #         loss_tensor = scalar_from_layer_output(self.keras_model, name, w)
    #         self.keras_model._losses.append(loss_tensor)

    #     reg_terms = [
    #         keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
    #         for w in self.keras_model.trainable_weights
    #         if 'gamma' not in w.name and 'beta' not in w.name
    #     ]
    #     if reg_terms:
    #         self.keras_model._losses.append(tf.add_n(reg_terms))

    #     self.keras_model.compile(
    #         optimizer=optimizer,
    #         loss=[None] * len(self.keras_model.outputs)
    #     )

    # def compile(self, learning_rate, momentum):
    #     # Optimizer
    #     optimizer = keras.optimizers.SGD(
    #         lr=learning_rate,
    #         momentum=momentum,
    #         clipnorm=self.config.GRADIENT_CLIP_NORM
    #     )

    #     # --- attach losses inside the graph (idempotent) ---
    #     if not getattr(self, "_losses_attached", False):
    #         # 1) tap the tensor(s) that used to feed your custom loss
    #         rank_src = self.keras_model.get_layer("rank_loss").output

    #         side = _AddScalarLoss(
    #             weight=self.config.LOSS_WEIGHTS.get("rank_loss", 1.0),
    #             name="rank_loss_tap"
    #         )(rank_src)

    #         # 2) add L2 as a per-batch loss (skip BN gamma/beta)
    #         side = _AddL2Tap(
    #             weights_to_reg=self.keras_model.trainable_weights,
    #             weight_decay=self.config.WEIGHT_DECAY,
    #             name="l2_reg_tap"
    #         )(side)

    #         # 3) keep outputs identical, just depend on the side path so it executes
    #         new_outputs = [
    #             _PassThrough(name=f"passthrough_{i}")([o, side])
    #             for i, o in enumerate(self.keras_model.outputs)
    #         ]
    #         self.keras_model = keras.Model(self.keras_model.inputs, new_outputs, name=self.keras_model.name)
    #         self._losses_attached = True

    #     # --- IMPORTANT: no model.add_loss or _losses.append anymore ---
    #     self.keras_model.compile(
    #         optimizer=optimizer,
    #         # loss=[None] * len(self.keras_model.outputs)
    #     )
    def compile(self, learning_rate, momentum):
        opt = keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum,
            clipnorm=self.config.GRADIENT_CLIP_NORM,
        )

        # loss_names = [
        #     "rpn_class_loss", "rpn_bbox_loss",
        #     "obj_sal_seg_class_loss", "obj_sal_seg_bbox_loss", "obj_sal_seg_mask_loss",
        # ]
        loss_names = ["rank_loss"]

        base_in  = self.keras_model.inputs
        base_out = list(self.keras_model.outputs)
        if not base_out:
            raise RuntimeError("Model has no outputs to anchor loss hooks to.")

        # Thread loss hooks onto an "anchor" that passes through unchanged.
        anchor = base_out[0]
        for name in loss_names:
            t = self.keras_model.get_layer(name).output
            w = float(self.config.LOSS_WEIGHTS.get(name, 1.0))
            anchor = LossHook(weight=w, name=f"{name}_hook")([t, anchor])

        # Keep all original heads visible; replace first head with the hooked anchor
        new_outputs = [anchor] + base_out[1:]
        model = keras.Model(base_in, new_outputs, name=self.keras_model.name)

        # L2 regularization via add_loss (materialize tensor)
        def l2_reg():
            terms = []
            for wv in model.trainable_weights:
                n = wv.name or ""
                if "gamma" in n or "beta" in n:
                    continue
                denom = tf.cast(tf.size(wv), tf.float32)
                terms.append(keras.regularizers.l2(self.config.WEIGHT_DECAY)(wv) / denom)
            return tf.add_n(terms) if terms else 0.0

        model.add_loss(l2_reg)

        # Important for option B: no per-output losses; we'll feed dummy targets matching structure.
        model.compile(optimizer=opt, loss=None, run_eagerly=False)

        # Expose how many outputs we expect so the input pipeline can create matching dummies.
        # Example dummy builder for tf.data:
        #   num = self.num_model_outputs
        #   train_ds = raw_ds.map(lambda x, y: (x, tuple(tf.constant(0.0) for _ in range(num))))
        self.num_model_outputs = len(new_outputs)

        self.keras_model = model




    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the corresponding Keras function with
            the addition of multi-GPU support and the ability to exclude
            some layers from loading.
            exclude: list of layer names to exclude
            """
        import h5py
        # Conditional import to support versions of Keras before 2.2
        # TODO: remove in about 6 months (end of 2018)
        try:
            from tensorflow.keras.saving.hdf5_format import load_weights_from_hdf5_group, load_weights_from_hdf5_group_by_name
        except Exception:
            try:
                from keras.saving.hdf5_format import load_weights_from_hdf5_group, load_weights_from_hdf5_group_by_name
            except Exception:
                from tensorflow.python.keras.saving.hdf5_format import load_weights_from_hdf5_group, load_weights_from_hdf5_group_by_name  # last resort (private API)

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            self.keras_model.load_weights(filepath, by_name=by_name, skip_mismatch=True)
        else:
            self.keras_model.load_weights(filepath, skip_mismatch=True)

        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    # Detection performed per single image
    def detect(self, input_data, verbose=0):

        assert self.mode == "inference", "Create model in inference mode."

        if verbose:
            log("Processing image")
            log("image", input_data[0])

        detections = self.keras_model.predict(input_data, verbose=0)

        return detections

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers
    


    def add_layer_output_loss(self, model, layer_name, weight=1.0):
        layer = model.get_layer(layer_name)
        outputs = nest.flatten(layer.output)  # handles Tensor / list / tuple
        # Reduce each to a scalar and sum them
        pieces = [tf.reduce_mean(tf.cast(t, tf.float32)) for t in outputs]
        loss_tensor = tf.add_n(pieces) * tf.cast(weight, tf.float32)
        model.add_loss(loss_tensor)

