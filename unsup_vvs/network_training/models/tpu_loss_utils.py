import tensorflow as tf
from models.loss_utils import get_cons_coefficient, mean_teacher_consitence_and_res, \
        instance_loss
import numpy as np
from models.rp_col_utils import pos2lbl
import pdb


def metric_fn(labels, logits, **kwargs):
    """Basic evaluation metric fn. Performed on CPU, do not reference TPU ops."""
    predictions = tf.argmax(logits, axis=1)
    precision_at_1 = tf.metrics.accuracy(labels, predictions)
    in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
    precision_at_5 = tf.metrics.mean(in_top_5)
    return {
        'top1': precision_at_1,
        'top5': precision_at_5}


def rp_metric_fn(labels, logits):

    print("validation logits:", logits)
    #num_cores = 8
    all_labels_ = []
    e_bs = tf.cast(logits.shape[0], tf.int32)
    for i in range(0, 12):
        labels_ = tf.expand_dims(pos2lbl(pair[i]), 0)
        labels_ = tf.reshape(labels_, [1, 1, 1])
        labels_ = tf.tile(labels_, [e_bs, 1, 1])
        all_labels_.append(labels_)
    
    all_labels = tf.concat(all_labels_, axis=1)

    predictions = tf.argmax(logits, axis=2)
    print("predictions shape:", predictions)

    precision_at_1 = tf.metrics.accuracy(all_labels, predictions)

    return {'top1': precision_at_1}


def depth_metric_fn(labels, logits):

    validation_loss = tf.metrics.mean_squared_error(labels, logits)
    
    print("depth validation loss:", validation_loss)

    return {
        'validation_loss': validation_loss
        }


def col_metric_fn(labels, logits):
    soft = True 
    if soft:
        labels = tf.argmax(labels, -1)

    logits = tf.reshape(logits, [-1, logits.get_shape().as_list()[-1]])
    print("validation logits:", logits)
    labels = tf.reshape(labels, [-1])
    print("validation labels:", labels)
    in_top_1 = tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32)
    
    logits = tf.argmax(logits, axis=1)
    precision_at_1_ = tf.metrics.accuracy(labels, logits)

    print("in_top_1:", in_top_1)
    precision_at_1 = tf.metrics.mean(in_top_1)

    return {'top1': precision_at_1,
            'top1_': precision_at_1_}


def combine_depth_imn_metric_fn(labels, logits):

    labels = tf.constant(
            [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], 
            [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    labels = tf.tile(labels, [8, 1])
    print("combine val labels:", labels)
    acc = tf.metrics.accuracy(
            labels=tf.argmax(labels, 1),
            predictions=tf.argmax(logits,1))
    print(acc)
    return {'validation_loss': acc}   


def tpu_instance_metric_fn(labels, logits, **kwargs):
    """
    Basic evaluation metric fn. Performed on CPU, do not reference TPU ops.
    """
    curr_dist = kwargs['i0']
    all_labels = kwargs['i1']
    _, top_indices = tf.nn.top_k(curr_dist, k=1)
    curr_pred = tf.gather(all_labels, tf.squeeze(top_indices, axis=1))
    precision_at_1 = tf.metrics.accuracy(labels, curr_pred)
    return {
        'top1': precision_at_1,}


def tpu_mean_teacher_metric_fn(labels, logits, **kwargs):
    """Basic evaluation metric fn. Performed on CPU, do not reference TPU ops."""
    class_logits = kwargs['i0']
    predictions = tf.argmax(class_logits, axis=1)
    precision_at_1 = tf.metrics.accuracy(labels, predictions)
    in_top_5 = tf.cast(tf.nn.in_top_k(class_logits, labels, 5), tf.float32)
    precision_at_5 = tf.metrics.mean(in_top_5)

    ema_logits = kwargs['i2']
    ema_predictions = tf.argmax(ema_logits, axis=1)
    ema_precision_at_1 = tf.metrics.accuracy(labels, ema_predictions)
    ema_in_top_5 = tf.cast(tf.nn.in_top_k(ema_logits, labels, 5), tf.float32)
    ema_precision_at_5 = tf.metrics.mean(ema_in_top_5)

    return {
        'top1': precision_at_1,
        'top5': precision_at_5,
        'ema_top1': ema_precision_at_1,
        'ema_top5': ema_precision_at_5,
        }


def combine_rp_imn_metric_fn(labels, logits):
    print("combine val:", labels)
    print("combine val:", logits)
    labels = tf.constant(
            [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], 
            [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    labels = tf.tile(labels, [8, 1])
    print("combine val labels:", labels)
    acc = tf.metrics.accuracy(
            labels=tf.argmax(labels, 1), 
            predictions=tf.argmax(logits,1))
    print(acc)
    return {'validation_loss': acc}  


def tpu_imagenet_loss(logits, labels, **kwargs):
    if isinstance(logits, dict):
        logits = logits[0]
    num_cat = logits.get_shape().as_list()[1]
    one_hot_labels = tf.one_hot(labels, num_cat)
    imnet_loss = tf.losses.softmax_cross_entropy(
            logits=logits, 
            onehot_labels=one_hot_labels)
    return imnet_loss


pair = [(0,1),(0,-1),(1,0),(-1,0),
        (1,1),(-1,-1),(1,-1),(-1,1),
        (1,0),(-1,0),(0,1),(0,-1)]


def tpu_rp_imagenet_loss(logits, labels, **kwargs):
    print("training loss logits:", logits)

    e_bs = tf.cast(logits.shape[0], tf.int32)
    all_labels = []
    for i in range(0, 12):
        one_hot_labels = tf.one_hot(pos2lbl(pair[i]), 8)
        one_hot_labels = tf.reshape(one_hot_labels, [1, 1, 8])
        one_hot_labels = tf.tile(one_hot_labels, [e_bs, 1, 1])
        all_labels.append(one_hot_labels)

    all_labels = tf.concat(all_labels, axis=1)
    print("training loss labels:", all_labels)

    imnet_loss = tf.losses.softmax_cross_entropy(
            logits=logits, onehot_labels=all_labels)

    return imnet_loss


def tpu_col_loss(logits, labels, **kwargs):
    soft = True
    print("input tpu_col_loss:", logits)
    flatten_logits = tf.reshape(logits, [-1, 313])
    #flatten_logits = tf.reshape(logits, [-1, 313])
    print("input label:", labels)
    flatten_labels = tf.reshape(labels, [-1, 313])

    loss = tf.losses.softmax_cross_entropy(
            logits=flatten_logits, onehot_labels=flatten_labels)
    return loss
            

def tpu_depth_loss(logits, labels, **kwargs):
    print("depth loss logits:", logits)
    print("depth loss labels:", labels)
    loss = tf.nn.l2_loss(logits - labels) / np.prod(labels.get_shape().as_list())
    return loss


def combine_depth_imn_loss(logits, labels, **kwargs):
    loss = tf.reduce_mean(logits)
    return loss


def combine_rp_imn_loss(logits, labels, **kwargs):
    loss = tf.reduce_mean(logits)
    return loss


def tpu_mean_teacher_loss(logits, labels, **kwargs):
    res_coef = kwargs.get('res_coef', 0.01)
    cons_ramp_len = kwargs.get('cons_ramp_len', 400000)
    cons_max_value = kwargs.get('cons_max_value', 10.0)
    mt_infant_loss = kwargs.get('mt_infant_loss', 0)

    cons_coefficient = get_cons_coefficient(cons_ramp_len, cons_max_value)

    # logits should be a dictionary having keys of 0 to 7
    class_logit = logits[0]

    # Get the classification loss first
    if mt_infant_loss == 0:
        one_hot_labels = tf.one_hot(labels, 1000)
    else:
        print("mt infant loss.")
        one_hot_labels = tf.one_hot(labels, 246)

    loss = tf.losses.softmax_cross_entropy(
            logits=class_logit, 
            onehot_labels=one_hot_labels)

    consistence_loss, res_loss = mean_teacher_consitence_and_res(
            class_logit=class_logit,
            cons_logit=logits[1],
            ema_class_logit=logits[2],
            cons_coefficient=cons_coefficient,
            res_coef=res_coef,
            )
    loss += consistence_loss + res_loss # Add all three losses

    # This part is for the unlabeled imagenet 
    consistence_loss, res_loss = mean_teacher_consitence_and_res(
            class_logit=logits[4],
            cons_logit=logits[5],
            ema_class_logit=logits[6],
            cons_coefficient=cons_coefficient,
            res_coef=res_coef,
            )
    loss += consistence_loss + res_loss # Add all three losses

    return loss


def get_tpu_loss_params(args):
    if args.tpu_task=='imagenet' \
            or args.tpu_task=='imagenet_rp' \
            or args.tpu_task=='color_tl':
        loss_func_pre = tpu_imagenet_loss

    if args.tpu_task in ['rp', 'rp_pbr', 'rp_only_pbr']:
        loss_func_pre = tpu_rp_imagenet_loss

    if args.tpu_task=='colorization' \
            or args.tpu_task=='color_ps' \
            or args.tpu_task=='color_pbr':
        loss_func_pre = tpu_col_loss
    if args.tpu_task=='depth' or args.tpu_task=='depth_pbr':
        loss_func_pre = tpu_depth_loss
    if args.tpu_task=='combine_depth_imn':
        loss_func_pre = combine_depth_imn_loss
    if args.tpu_task=='combine_rp_imn' \
            or args.tpu_task=='combine_rp_col' \
            or args.tpu_task=='combine_rdc' \
            or args.tpu_task=='combine_rdc_imn' \
            or args.tpu_task=='combine_rp_col_ps' \
            or args.tpu_task=='combine_rci':
        loss_func_pre = combine_rp_imn_loss
    if args.tpu_task=='mean_teacher':
        loss_func_pre = tpu_mean_teacher_loss
    if args.tpu_task=='instance_task':
        def _instance_loss(logits, labels, **kwargs):
            loss_pure, _, _ = instance_loss(
                    logits[0], logits[1], 
                    instance_k=args.instance_k,
                    instance_data_len=args.instance_data_len,)
            return loss_pure
        loss_func_pre = _instance_loss
    if args.tpu_task=='cpc':
        def _cpc_loss(logits, labels, **kwargs):
            loss = tf.reduce_mean(logits)
            return loss
        loss_func_pre = _cpc_loss
    if args.tpu_task=='multi_imagenet':
        def _multi_imagenet_loss(logits, labels, **kwargs):
            all_logits = tf.unstack(logits, axis=1)
            loss = sum([tpu_imagenet_loss(each_logit, labels, **kwargs) \
                        for each_logit in all_logits])
            return loss
        loss_func_pre = _multi_imagenet_loss

    if args.tpu_task:
        def _wrapper_wd(*args, **kwargs):
            loss_pure = loss_func_pre(*args, **kwargs)
            reg_losses = tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES)
            if len(reg_losses)!=0:
                loss_all = tf.add(loss_pure, tf.reduce_sum(reg_losses))
            else:
                loss_all = loss_pure
            return loss_all
        loss_func = _wrapper_wd

    loss_params = {
            'agg_func': tf.reduce_mean,
            'loss_per_case_func': loss_func,
            'loss_per_case_func_params': {},
            'loss_func_kwargs': {},
            }
    if args.tpu_task in ['cpc', 'multi_imagenet']:
        loss_params['inputs_as_dict'] = True
    if args.tpu_task=='mean_teacher':
        loss_params['loss_func_kwargs'] = {
                'res_coef':args.res_coef,
                'cons_ramp_len':args.cons_ramp_len,
                'cons_max_value':args.cons_max_value,
                'mt_infant_loss':args.mt_infant_loss,
                }
    return loss_params
