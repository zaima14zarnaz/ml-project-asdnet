import os
import re
import datetime
import keras
import multiprocessing
from obj_sal_seg_branch import Model_Sal_Seg
from obj_sal_seg_branch import DataGenerator
from obj_sal_seg_branch.ObjSegMaskConfig import ObjSegMaskConfig
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from fpn_network import model_utils
from fpn_network import utils


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


class SOSNet():
    def __init__(self, mode, config, model_dir):
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()

        self.keras_model = Model_Sal_Seg.build_saliency_seg_model(config, mode)

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

        # TODO: Update
        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/[\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/sos\_net\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Create log_dir if not exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "sos_net_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
            augmentation=None, custom_callbacks=None):

        assert self.mode == "training", "Create model in training mode."

        layer_regex = {
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Original generators (likely Python generators / Keras Sequence)
        train_generator = DataGenerator.data_generator(
            train_dataset, self.config, shuffle=True,
            augmentation=augmentation, batch_size=self.config.BATCH_SIZE)
        print('Generated training data')

        val_generator = DataGenerator.data_generator(
            val_dataset, self.config, shuffle=True,
            batch_size=self.config.BATCH_SIZE)
        print('Generated val data')

        logs_path = self.log_dir + "/training.log"
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=1, save_weights_only=True),
            keras.callbacks.CSVLogger(logs_path, separator=",", append=True),
        ]
        if custom_callbacks:
            callbacks += custom_callbacks
        print('Created callbacks')

        # Compile (Option B): keeps heads visible; sets self.num_model_outputs
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)
        print('Compiled SOSNet')

        num = self.num_model_outputs

        # --- Wrap generators to add dummy targets matching the model's output structure ---
        def wrap_with_dummies(gen, num_outputs):
            # gen yields either (x, y) or just x; we ignore y and add a tuple of dummy targets
            for batch in gen:
                if isinstance(batch, tuple) and len(batch) == 2:
                    x, _ = batch
                else:
                    x = batch
                # Use scalar float32 dummies; shapes don't matter since loss=None
                dummies = tuple(tf.constant(0.0, dtype=tf.float32) for _ in range(num_outputs))
                yield x, dummies

        train_gen = wrap_with_dummies(train_generator, num)
        val_gen   = wrap_with_dummies(val_generator,   num)

        # Workers: fix comparison; keep single-threaded if you’re debugging
        if os.name == 'nt':
            workers = 0
        else:
            workers = 0  # or multiprocessing.cpu_count() if you want parallelism

        # Train with wrapped generators
        self.keras_model.fit(
            train_generator,
            validation_data=val_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            validation_steps=self.config.VALIDATION_STEPS,
            callbacks=callbacks,
            workers=workers,
            use_multiprocessing=True,
            max_queue_size=10,
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

    #     # Compile SGD
    #     # optimizer = keras.optimizers.SGD(
    #     #     lr=learning_rate, momentum=momentum,
    #     #     clipnorm=self.config.GRADIENT_CLIP_NORM)

    #     # # Add Losses
    #     # # First, clear previously set losses to avoid duplication
    #     # self.keras_model._losses = []
    #     # self.keras_model._per_input_losses = {}
    #     # loss_names = [
    #     #     "rpn_class_loss", "rpn_bbox_loss",
    #     #     "obj_sal_seg_class_loss", "obj_sal_seg_bbox_loss", "obj_sal_seg_mask_loss"]
    #     # # for name in loss_names:
    #     # #     layer = self.keras_model.get_layer(name)
    #     # #     if layer.output in self.keras_model.losses:
    #     # #         continue
    #     # #     loss = (tf.reduce_mean(layer.output, keepdims=True)
    #     # #             * self.config.LOSS_WEIGHTS.get(name, 1.))
    #     # #     self.keras_model.add_loss(loss)
    #     # for name in loss_names:
    #     #     w = self.config.LOSS_WEIGHTS.get(name, 1.0)
    #     #     t = self.keras_model.get_layer(name).output  # may be tensor or structure
    #     #     # Wrap with AddLoss to register as model loss + metric safely
    #     #     _ = SOSNet.AddLoss(weight=w, metric_name=name + "_mean", name=name + "_addloss")(t)
    #     # # Add L2 Regularization
    #     # # Skip gamma and beta weights of batch normalization layers.
    #     # # reg_losses = [
    #     # #     keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
    #     # #     for w in self.keras_model.trainable_weights
    #     # #     if 'gamma' not in w.name and 'beta' not in w.name]
    #     # # self.keras_model.add_loss(tf.add_n(reg_losses))
    #     # reg_losses = []
    #     # for w in self.keras_model.trainable_weights:
    #     #     n = w.name or ""
    #     #     if "gamma" in n or "beta" in n:
    #     #         continue
    #     #     # per-weight scaled L2; divide by number of elements to keep magnitude stable
    #     #     denom = tf.cast(tf.size(w), tf.float32)
    #     #     reg_losses.append(keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / denom)

    #     # # safer: handle empty list + defer computation
    #     # self.keras_model.add_loss(lambda: tf.add_n(reg_losses) if reg_losses else 0.0)


    #     # Compile
    #     # self.keras_model.compile(
    #     #     optimizer=optimizer,
    #     #     loss=[None] * len(self.keras_model.outputs))

    #     # Add metrics for losses
    #     # for name in loss_names:
    #     #     if name in self.keras_model.metrics_names:
    #     #         continue
    #     #     layer = self.keras_model.get_layer(name)
    #     #     self.keras_model.metrics_names.append(name)
    #     #     loss = (tf.reduce_mean(layer.output, keepdims=True)
    #     #             * self.config.LOSS_WEIGHTS.get(name, 1.))
    #     #     self.keras_model.metrics_tensors.append(loss)
    #     optimizer = keras.optimizers.SGD(
    #         learning_rate=learning_rate,
    #         momentum=momentum,
    #         clipnorm=self.config.GRADIENT_CLIP_NORM
    #     )

    #     loss_names = [
    #         "rpn_class_loss", "rpn_bbox_loss",
    #         "obj_sal_seg_class_loss", "obj_sal_seg_bbox_loss", "obj_sal_seg_mask_loss"
    #     ]

    #     for name in loss_names:
    #         w = self.config.LOSS_WEIGHTS.get(name, 1.0)
    #         t = self.keras_model.get_layer(name).output
    #         _ = SOSNet.AddLoss(weight=w, metric_name=name, name=f"{name}_addloss")(t)

    #     # L2 regularization
    #     reg_losses = []
    #     for w in self.keras_model.trainable_weights:
    #         n = w.name or ""
    #         if "gamma" in n or "beta" in n:
    #             continue
    #         denom = tf.cast(tf.size(w), tf.float32)
    #         reg_losses.append(keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / denom)

    #     self.keras_model.add_loss(lambda: tf.add_n(reg_losses) if reg_losses else 0.0)

    #     # Compile (start with eager for debugging; turn off later)
    #     self.keras_model.compile(optimizer=optimizer, run_eagerly=True)
    # def compile(self, learning_rate, momentum):
    #     opt = keras.optimizers.SGD(
    #         learning_rate=learning_rate,
    #         momentum=momentum,
    #         clipnorm=self.config.GRADIENT_CLIP_NORM,
    #     )

    #     base_in  = self.keras_model.inputs
    #     base_out = list(self.keras_model.outputs)

    #     loss_names = [
    #         "rpn_class_loss", "rpn_bbox_loss",
    #         "obj_sal_seg_class_loss", "obj_sal_seg_bbox_loss", "obj_sal_seg_mask_loss",
    #     ]

    #     loss_outputs   = []
    #     loss_weights   = []
    #     for name in loss_names:
    #         t = self.keras_model.get_layer(name).output            # may be tuple/list
    #         parts = [tf.cast(p, tf.float32) for p in tf.nest.flatten(t)]
    #         # combine to a single scalar per batch
    #         loss_tensor = tf.add_n([tf.reduce_mean(p) for p in parts])
    #         loss_outputs.append(loss_tensor)
    #         loss_weights.append(float(self.config.LOSS_WEIGHTS.get(name, 1.0)))

    #     # Rebuild model with extra outputs (task outputs + loss outputs)
    #     model = keras.Model(base_in, base_out + loss_outputs, name=self.keras_model.name)

    #     # Regularization as a model loss (callable)
    #     def l2_reg():
    #         terms = []
    #         for wv in model.trainable_weights:
    #             n = wv.name or ""
    #             if "gamma" in n or "beta" in n:
    #                 continue
    #             denom = tf.cast(tf.size(wv), tf.float32)
    #             terms.append(keras.regularizers.l2(self.config.WEIGHT_DECAY)(wv) / denom)
    #         return tf.add_n(terms) if terms else 0.0
    #     model.add_loss(l2_reg)

    #     # Build loss lists: your task losses for original outputs + identity for added loss outputs
    #     def identity_loss(y_true, y_pred):
    #         return tf.reduce_mean(y_pred)

    #     task_losses = [None] * len(base_out)   # if your main heads already produce losses internally
    #     # If you actually have supervised outputs, replace Nones with real loss fns (e.g., "categorical_crossentropy")

    #     model.compile(
    #         optimizer=opt,
    #         loss=task_losses + [identity_loss] * len(loss_outputs),
    #         loss_weights=[1.0] * len(base_out) + loss_weights,
    #         run_eagerly=True,   # keep this on until stable; switch off later for speed
    #     )

    #     self.keras_model = model
    def compile(self, learning_rate, momentum):
        opt = keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum,
            clipnorm=self.config.GRADIENT_CLIP_NORM,
        )

        loss_names = [
            "rpn_class_loss", "rpn_bbox_loss",
            "obj_sal_seg_class_loss", "obj_sal_seg_bbox_loss", "obj_sal_seg_mask_loss",
        ]

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
            from keras.engine import saving
        except ImportError:
            # Keras before 2.2 used the 'topology' namespace.
            from keras.engine import topology as saving

        # if exclude:
        #     by_name = True

        # if h5py is None:
        #     raise ImportError('`load_weights` requires h5py.')
        # f = h5py.File(filepath, mode='r')
        # if 'layer_names' not in f.attrs and 'model_weights' in f:
        #     f = f['model_weights']

        # # In multi-GPU training, we wrap the model. Get layers
        # # of the inner model because they have the weights.
        # keras_model = self.keras_model
        # layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
        #     else keras_model.layers

        # # Exclude some layers
        # if exclude:
        #     layers = filter(lambda l: l.name not in exclude, layers)

        # # if by_name:
        # #     saving.load_weights_from_hdf5_group_by_name(f, layers)
        # # else:
        # #     saving.load_weights_from_hdf5_group(f, layers)
        # keras_model.load_weights(f, by_name=True, skip_mismatch=True)
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") else keras_model.layers

        if exclude:
            by_name = True
            # cache current weights of excluded layers
            excluded_cache = {l.name: l.get_weights() for l in layers if l.name in exclude}

            # load by name, skip mismatches (handles missing keys, shape diffs, etc.)
            keras_model.load_weights(filepath, by_name=True, skip_mismatch=True)

            # restore excluded layers
            for l in layers:
                if l.name in exclude and l.name in excluded_cache:
                    if l.get_weights():  # layer has weights
                        l.set_weights(excluded_cache[l.name])
        else:
            # no exclusions: just use standard loader
            if by_name:
                keras_model.load_weights(filepath, by_name=True, skip_mismatch=True)
            else:
                keras_model.load_weights(filepath)


        # if hasattr(f, 'close'):
        #     f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    # Detection performed per single image
    def detect(self, images, verbose=0):
        """Runs the detection pipeline.

             images: List of images, potentially of different sizes.

             Returns a list of dicts, one dict per image. The dict contains:
             rois: [N, (y1, x1, y2, x2)] detection bounding boxes
             class_ids: [N] int class IDs
             scores: [N] float probability scores for the class IDs
             masks: [H, W, N] instance binary masks
             """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(
            images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, \
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)

        # Run detection
        detections, _, _, mrcnn_mask, _, _, _ = \
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)

        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks = \
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matrices [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matrices:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = model_utils.mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = model_utils.compose_image_meta(
                0, image.shape, molded_image.shape, window, scale,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, original_image_shape,
                          image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        mrcnn_mask: [N, height, width, num_classes]
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                image is excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1) \
            if full_masks else np.empty(original_image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks

    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = model_utils.compute_backbone_shapes(self.config, image_shape)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]
    
    class AddLoss(keras.layers.Layer):
        def __init__(self, weight=1.0, metric_name=None, **kw):
            super().__init__(**kw)
            self.weight = float(weight)
            self.metric_name = metric_name

        def call(self, x):
            parts = tf.nest.flatten(x)
            parts = [tf.cast(p, tf.float32) for p in parts]
            loss_tensor = tf.add_n([tf.reduce_mean(p) for p in parts]) * self.weight
            self.add_loss(loss_tensor)
            if self.metric_name:
                # log the unweighted mean; change to loss_tensor if you prefer weighted
                self.add_metric(tf.add_n([tf.reduce_mean(p) for p in parts]),
                                name=self.metric_name, aggregation='mean')
            return x

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