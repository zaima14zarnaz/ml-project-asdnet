from keras.models import *
from keras.layers import *
from tensorflow.keras import backend as K
from fpn_network.BatchNorm import BatchNorm
from model import SaliencyRankClass, Losses
from model.AttentionLayer import AttentionLayer
from tensorflow.keras.layers import TimeDistributed

from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, BatchNormalization as BatchNorm, Activation,
    GlobalAveragePooling2D, TimeDistributed, Concatenate, Lambda
)





# def build_saliency_rank_model(config, mode):
#     # *********************** INPUTS ***********************
#     input_obj_features = Input(shape=(config.SAL_OBJ_NUM, 1, 1, config.OBJ_FEAT_SIZE), name="input_obj_feat")
#     input_obj_spatial_masks = Input(shape=(config.SAL_OBJ_NUM, 32, 32, 1), name="input_obj_spatial_masks")
#     input_P5_feat = Input(shape=(32, 32, 256), name="input_P5_feat")

#     if mode == "training":
#         input_target_rank = Input(shape=(config.SAL_OBJ_NUM,), name="input_gt_ranks")

#     # *********************** PROCESS Image/P5 FEATURES ***********************
#     img_feat = Conv2D(config.BOTTLE_NECK_SIZE, (3, 3), name="img_feat_conv_1")(input_P5_feat)
#     img_feat = BatchNorm(name="img_feat_bn_1")(img_feat, training=config.TRAIN_BN)
#     img_feat = Activation('relu')(img_feat)

#     img_feat = GlobalAveragePooling2D()(img_feat)

#     # *********************** SELECTIVE ATTENTION MODULE ***********************
#     # Reduce dimension to BOTTLNECK
#     obj_feature = TimeDistributed(Conv2D(config.BOTTLE_NECK_SIZE, (1, 1)), name="obj_feat_reduce_conv1")(input_obj_features)
#     obj_feature = TimeDistributed(BatchNorm(), name='obj_feat_reduce_bn1')(obj_feature, training=config.TRAIN_BN)
#     obj_feature = Activation('relu')(obj_feature)

#     obj_feature = Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2), name="obj_feat_squeeze")(obj_feature)

#     sa_feat = selective_attention_module(config.NUM_ATTN_HEADS, obj_feature, img_feat, config)

#     # *********************** OBJECT SPATIAL MASK MODULE ***********************
#     spatial_mask_feat = object_spatial_mask_module(input_obj_spatial_masks, config)

#     # CONCATENATE OBJ_FEAT_MASKS + OBJ_SPATIAL_MASKS
#     obj_feature = Concatenate()([sa_feat, spatial_mask_feat])

#     # *********************** FINAL OBJECT FEATURE ***********************
#     # FC layer for reducing the attention features
#     final_obj_feat = TimeDistributed(Dense(config.RANK_FEAT_SIZE), name="obj_final_feat_dense_1")(obj_feature)
#     final_obj_feat = TimeDistributed(BatchNorm(), name='obj_final_feat_bn_1')(final_obj_feat, training=config.TRAIN_BN)
#     final_obj_feat = Activation('relu')(final_obj_feat)

#     # *********************** OBJECT RANK ORDER NETWORK ***********************
#     # Ranking Network
#     point_scoring_model = SaliencyRankClass.build_rank_class_model(config)

#     # <- Insert autoregressive decoder here
#     # Perform Ranking
#     # object_rank = point_scoring_model(final_obj_feat)
#     # object_rank = TimeDistributed(point_scoring_model, name="td_point_score")(final_obj_feat)
#     B = tf.shape(final_obj_feat)[0]
#     R = tf.shape(final_obj_feat)[1]
#     F = tf.shape(final_obj_feat)[2]  # 512

#     feat2d = tf.reshape(final_obj_feat, (-1, F))         # (B*R, 512)
#     scores2d = point_scoring_model(feat2d)               # (B*R, out_dim)
#     object_rank = tf.reshape(scores2d, (B, R, -1))       # (B, R, out_dim)


#     if mode == "training":
#         # *********************** LOSS **********************
#         # Rank Loss
#         rank_loss = Lambda(lambda x: Losses.sparse_categorical_cross_entropy_pos_contrib(*x), name="rank_loss")(
#             [input_target_rank, object_rank])

#         # *********************** FINAL ***********************
#         # Model
#         inputs = [input_obj_features, input_obj_spatial_masks, input_P5_feat,
#                   input_target_rank]
#         outputs = [object_rank, rank_loss]
#         model = Model(inputs=inputs, outputs=outputs, name="attn_shift_saliency_rank_model")
#     else:
#         # *********************** FINAL ***********************
#         # Model
#         inputs = [input_obj_features, input_obj_spatial_masks,
#                   input_P5_feat]
#         outputs = [object_rank]
#         model = Model(inputs=inputs, outputs=outputs, name="attn_shift_saliency_rank_model")

#     return model

def build_saliency_rank_model(config, mode):
    # *********************** INPUTS ***********************
    input_obj_features = Input(shape=(config.SAL_OBJ_NUM, 1, 1, config.OBJ_FEAT_SIZE), name="input_obj_feat")
    input_obj_spatial_masks = Input(shape=(config.SAL_OBJ_NUM, 32, 32, 1), name="input_obj_spatial_masks")
    input_P5_feat = Input(shape=(32, 32, 256), name="input_P5_feat")

    


    # *********************** PROCESS Image/P5 FEATURES ***********************
    img_feat = Conv2D(config.BOTTLE_NECK_SIZE, (3, 3), name="img_feat_conv_1")(input_P5_feat)
    img_feat = BatchNorm(name="img_feat_bn_1")(img_feat, training=config.TRAIN_BN)
    img_feat = Activation('relu')(img_feat)

    img_feat = GlobalAveragePooling2D()(img_feat)

    # *********************** SELECTIVE ATTENTION MODULE ***********************
    # Reduce dimension to BOTTLNECK
    obj_feature = TimeDistributed(Conv2D(config.BOTTLE_NECK_SIZE, (1, 1)), name="obj_feat_reduce_conv1")(input_obj_features)
    obj_feature = TimeDistributed(BatchNorm(), name='obj_feat_reduce_bn1')(obj_feature, training=config.TRAIN_BN)
    obj_feature = Activation('relu')(obj_feature)

    obj_feature = Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2), name="obj_feat_squeeze")(obj_feature)

    sa_feat = selective_attention_module(config.NUM_ATTN_HEADS, obj_feature, img_feat, config)

    # *********************** OBJECT SPATIAL MASK MODULE ***********************
    spatial_mask_feat = object_spatial_mask_module(input_obj_spatial_masks, config)

    # CONCATENATE OBJ_FEAT_MASKS + OBJ_SPATIAL_MASKS
    obj_feature = Concatenate()([sa_feat, spatial_mask_feat])

    # *********************** FINAL OBJECT FEATURE ***********************
    # FC layer for reducing the attention features
    final_obj_feat = TimeDistributed(Dense(config.RANK_FEAT_SIZE), name="obj_final_feat_dense_1")(obj_feature)
    final_obj_feat = TimeDistributed(BatchNorm(), name='obj_final_feat_bn_1')(final_obj_feat, training=config.TRAIN_BN)
    final_obj_feat = Activation('relu')(final_obj_feat)

    # *********************** OBJECT RANK ORDER NETWORK ***********************
    # Ranking Network
    point_scoring_model = SaliencyRankClass.build_rank_class_model(config)
    # mha = layers.MultiHeadAttention(num_heads=config.NUM_ATTN_HEADS, 
                                    # key_dim=final_obj_feat.shape[-1])
    # object_rank, rank_order = autoregressive_decoder(final_obj_feat, point_scoring_model, mha, num_steps=5)
    if mode == "training":
        input_target_rank = Input(shape=(config.SAL_OBJ_NUM,), name="input_gt_ranks")

        # ---- derive gt_rank_order inside the graph ----
        # argsort: smaller rank value = higher importance
        gt_rank_order = Lambda(lambda x: tf.argsort(x, axis=-1))(input_target_rank)   # (B, SAL_OBJ_NUM)
        gt_rank_order = Lambda(lambda x: x[:, :config.TOP_K])(gt_rank_order)          # (B, TOP_K)

        # Teacher-forced autoregressive decoding
        object_rank, rank_order = autoregressive_decoder(
            final_obj_feat,
            point_scoring_model,
            config,
            num_steps=config.TOP_K,
            gt_rank_order=gt_rank_order,
            training=True
        )
    else:
        object_rank, rank_order = autoregressive_decoder(
            final_obj_feat,
            point_scoring_model,
            config,
            num_steps=config.TOP_K,
            training=False
        )



    # <- Insert autoregressive decoder here
    # Perform Ranking
    # object_rank = point_scoring_model(final_obj_feat)
    # object_rank = TimeDistributed(point_scoring_model, name="td_point_score")(final_obj_feat)
    # B = tf.shape(final_obj_feat)[0]
    # R = tf.shape(final_obj_feat)[1]
    # F = tf.shape(final_obj_feat)[2]  # 512

    # feat2d = tf.reshape(final_obj_feat, (-1, F))         # (B*R, 512)
    # scores2d = point_scoring_model(feat2d)               # (B*R, out_dim)
    # object_rank = tf.reshape(scores2d, (B, R, -1))       # (B, R, out_dim)




    if mode == "training":
        # *********************** LOSS **********************
        # object_rank: (B, S, R, C) from autoregressive_decoder
        # gt_rank_order: (B, S) ground-truth indices sequence (teacher forcing)

        def stepwise_ce_loss(args):
            gt_order, pred = args  # gt_order: (B,S), pred: (B,S,R,C)
            B = tf.shape(pred)[0]
            S = tf.shape(pred)[1]
            R = tf.shape(pred)[2]

            losses = []
            for t in range(config.TOP_K):  # loop over steps
                step_pred = pred[:, t, :, :]           # (B,R,C)
                step_logits = tf.reduce_max(step_pred, axis=-1)  # (B,R)
                step_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=gt_order[:, t], logits=step_logits
                )  # (B,)
                losses.append(step_loss)

            losses = tf.stack(losses, axis=1)   # (B,S)
            return tf.reduce_mean(losses)       # scalar

        rank_loss = Lambda(stepwise_ce_loss, name="rank_loss")([gt_rank_order, object_rank])

        # For inspection/metrics: average predictions across steps
        object_rank_mean = tf.reduce_mean(object_rank, axis=1)  # (B,R,C)

        # *********************** FINAL ***********************
        inputs = [input_obj_features, input_obj_spatial_masks, input_P5_feat,
                input_target_rank]  # still 4 inputs if gt_rank_order is derived inside
        outputs = [object_rank_mean, rank_loss]
        model = Model(inputs=inputs, outputs=outputs, name="attn_shift_saliency_rank_model")
    else:
        inputs = [input_obj_features, input_obj_spatial_masks, input_P5_feat]
        outputs = [object_rank]
        model = Model(inputs=inputs, outputs=outputs, name="attn_shift_saliency_rank_model")



    return model



def object_spatial_mask_module(in_obj_spatial_masks, config):
    # *********************** OBJECT SPATIAL MASKS ***********************
    obj_spa_mask = TimeDistributed(Conv2D(96, (5, 5), strides=2, padding="same"), name="obj_spatial_mask_conv_1")(
        in_obj_spatial_masks)
    obj_spa_mask = TimeDistributed(BatchNorm(), name='obj_spatial_mask_bn_1')(obj_spa_mask, training=config.TRAIN_BN)
    obj_spa_mask = Activation("relu")(obj_spa_mask)

    obj_spa_mask = TimeDistributed(Conv2D(128, (5, 5), strides=2, padding="same"), name="obj_spatial_mask_conv_2")(
        obj_spa_mask)
    obj_spa_mask = TimeDistributed(BatchNorm(), name='obj_spatial_mask_bn_2')(obj_spa_mask, training=config.TRAIN_BN)
    obj_spa_mask = Activation("relu")(obj_spa_mask)

    obj_spa_mask = TimeDistributed(Conv2D(64, (8, 8)), name="obj_spatial_mask_conv_3")(obj_spa_mask)
    obj_spa_mask = TimeDistributed(BatchNorm(), name='obj_spatial_mask_bn_3')(obj_spa_mask, training=config.TRAIN_BN)
    obj_spa_mask = Activation("relu")(obj_spa_mask)

    obj_spatial_mask_feat = Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2), name="obj_spatial_mask_squeeze")(obj_spa_mask)

    return obj_spatial_mask_feat


def selective_attention_module(num_heads, obj_feat, img_feat, config):
    head_outputs = []
    for h in range(num_heads):
        theta_dense_name = "head_" + str(h) + "_obj_theta_dense_1"
        theta_bn_name = "head_" + str(h) + "_obj_theta_bn_1"
        phi_dense_name = "head_" + str(h) + "_img_phi_dense_1"
        phi_bn_name = "head_" + str(h) + "_img_phi_bn_1"
        g_dense_name = "head_" + str(h) + "_img_g_dense_1"
        g_bn_name = "head_" + str(h) + "_img_g_bn_1"

        proj_feat_size = config.BOTTLE_NECK_SIZE // num_heads

        # Project features
        obj_theta = TimeDistributed(Dense(proj_feat_size), name=theta_dense_name)(obj_feat)
        obj_theta = TimeDistributed(BatchNorm(), name=theta_bn_name)(obj_theta, training=config.TRAIN_BN)
        img_phi = Dense(proj_feat_size, name=phi_dense_name)(img_feat)
        img_phi = BatchNorm(name=phi_bn_name)(img_phi, training=config.TRAIN_BN)
        img_g = Dense(proj_feat_size, name=g_dense_name)(img_feat)
        img_g = BatchNorm(name=g_bn_name)(img_g, training=config.TRAIN_BN)

        # Repeat Vectors
        img_phi = RepeatVector(config.SAL_OBJ_NUM)(img_phi)
        img_g = RepeatVector(config.SAL_OBJ_NUM)(img_g)

        attn_name = "attn_layer_" + str(h)
        attn = AttentionLayer(config, name=attn_name)([obj_theta, img_phi, img_g])

        head_outputs.append(attn)

    final_attn = Concatenate()(head_outputs) if num_heads > 1 else head_outputs[0]

    # Linear
    final_attn = TimeDistributed(Dense(config.BOTTLE_NECK_SIZE), name='obj_attn_feat_dense_1')(final_attn)
    final_attn = TimeDistributed(BatchNorm(), name='obj_attn_feat_bn_1')(final_attn, training=config.TRAIN_BN)
    final_attn = Activation('relu')(final_attn)

    # Add Residual
    final_attn = Add()([final_attn, obj_feat])
    final_attn = Activation('relu')(final_attn)

    # Feed_forward
    final_obj_feat = TimeDistributed(Dense(config.BOTTLE_NECK_SIZE), name="obj_attn_feat_ff_dense_1")(final_attn)
    final_obj_feat = TimeDistributed(BatchNorm(), name='obj_attn_feat_ff_bn_1')(final_obj_feat, training=config.TRAIN_BN)
    final_obj_feat = Activation('relu')(final_obj_feat)

    return final_obj_feat

def autoregressive_decoder(final_obj_feat, point_scoring_model, config,
                           num_steps=5, gt_rank_order=None, training=False):
    B = tf.shape(final_obj_feat)[0]
    R = tf.shape(final_obj_feat)[1]
    Fdim = tf.shape(final_obj_feat)[2]
    Fdim_int = int(final_obj_feat.shape[-1]) or config.RANK_FEAT_SIZE

    block_no = 8 # experiment with 2-8 blocks

    # Layers
    history_ln = layers.LayerNormalization(name="history_self_attn_ln")
    self_mha = layers.MultiHeadAttention(num_heads=config.NUM_ATTN_HEADS, key_dim=Fdim_int, name="history_self_attn")
    self_att_stack = [
        layers.MultiHeadAttention(num_heads=config.NUM_ATTN_HEADS, key_dim=Fdim_int, name=f"self_attn_{i+1}")
        for i in range(block_no-1)
    ]
    ln1_stack = [layers.LayerNormalization(name=f"self_attn_ln1_{i+1}") for i in range(block_no-1)]
    self_attn_ffn_stack = [
        tf.keras.Sequential([
            layers.Dense(Fdim_int, activation="gelu"),
            layers.Dense(Fdim_int)
        ], name=f"self_attn_ffn_{i+1}")
        for i in range(block_no-1)
    ]
    ln2_stack = [layers.LayerNormalization(name=f"self_attn_ln2_{i+1}") for i in range(block_no-1)]
    mha_stack = [
        layers.MultiHeadAttention(num_heads=config.NUM_ATTN_HEADS, key_dim=Fdim_int, name=f"cross_attn_{i+1}")
        for i in range(block_no)
    ]
    ln3_stack = [layers.LayerNormalization(name=f"cross_attn_ln2_{i+1}") for i in range(block_no)]
    cross_attn_ffn_stack = [
        tf.keras.Sequential([
            layers.Dense(Fdim_int, activation="gelu"),
            layers.Dense(Fdim_int)
        ], name=f"cross_attn_ffn_{i+1}")
        for i in range(block_no)
    ]
    ln4_stack = [layers.LayerNormalization(name=f"cross_attn_ln3_{i+1}") for i in range(block_no)]

    ranked_indices, ranked_scores = [], []
    history_feats = []  # store features of previously ranked objects
    mask = tf.zeros((B, R), dtype=tf.bool)

    for t in range(num_steps):
        if t == 0:
            # No history yet → score directly from final_obj_feat
            feat2d = tf.reshape(final_obj_feat, (-1, Fdim))
            scores2d = point_scoring_model(feat2d)
            scores = tf.reshape(scores2d, (B, R, -1))
        else:
            if len(history_feats) == 1:
                # One history object → use it directly as query
                query_hist = tf.expand_dims(history_feats[-1], 1)   # (B,1,F)
            else:
                # Cap history at last 2 objects
                hist_stack = tf.stack(history_feats[-min(3,len(history_feats)):], axis=1)   # (B,2,F)

                # Self-attention across the 2 history objects
                attn_hist = self_mha(query=hist_stack,
                                    value=hist_stack,
                                    key=hist_stack)                # (B,2,F)

                # Residual + LayerNorm
                attn_hist = history_ln(hist_stack + attn_hist)      # (B,2,F)

                # Pool into single query representation
                query_hist = tf.reduce_mean(attn_hist, axis=1, keepdims=True)  # (B,1,F)

            # --- Transformer-style block stack (4 blocks) ---
            attn_in = query_hist
            for i in range(block_no):
                # Model variation 2 (self attention block inserted in decoder block before every cross attention sub-block)
                # if i != 0:
                #     # (1) self-attention on the history query itself
                #     sa_out = self_att_stack[i-1](query=attn_in, value=attn_in, key=attn_in)   # (B,1,F)
                #     attn_in = ln1_stack[i-1](attn_in + sa_out)

                # (3) cross-attention with object features
                ca_out = mha_stack[i](query=attn_in, value=final_obj_feat, key=final_obj_feat)
                attn_in = ln3_stack[i](attn_in + ca_out)  # residual + LN

                # (4) FFN after cross-attn
                ffn_out = cross_attn_ffn_stack[i](attn_in)
                attn_in = ln4_stack[i](attn_in + ffn_out)

            attn_out = tf.squeeze(attn_in, axis=1)            # (B,F)

            expanded = tf.tile(tf.expand_dims(attn_out, 1), [1, R, 1])  # (B,R,F)
            feat2d = tf.reshape(expanded, (-1, Fdim))
            scores2d = point_scoring_model(feat2d)
            scores = tf.reshape(scores2d, (B, R, -1))

        # Mask already selected objects
        logits = tf.reduce_max(scores, axis=-1)
        logits = tf.where(mask, tf.cast(-1e9, logits.dtype), logits)

        # Pick index (teacher forcing or prediction)
        if training and gt_rank_order is not None:
            top_idx = gt_rank_order[:, t]
        else:
            top_idx = tf.argmax(logits, axis=-1)

        ranked_indices.append(top_idx)
        ranked_scores.append(scores)

        # Update mask
        mask = mask | tf.one_hot(top_idx, R, on_value=True, off_value=False, dtype=tf.bool)

        # Store the feature of the selected object for history
        selected_feat = tf.gather(final_obj_feat, top_idx, batch_dims=1)  # (B,F)
        history_feats.append(selected_feat)

    return tf.stack(ranked_scores, axis=1), tf.stack(ranked_indices, axis=1)
