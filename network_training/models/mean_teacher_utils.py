import tensorflow as tf
from contextlib import contextmanager
import numpy as np


def rampup_rampdown_lr(
        init_lr, target_lr, 
        nb_per_epoch, global_step,
        enable_ramp_down=False,
        ramp_up_epoch=2, ramp_down_epoch=75, 
        ):
    curr_epoch  = tf.div(
            tf.cast(global_step, tf.float32), 
            tf.cast(nb_per_epoch, tf.float32))
    ramp_up_epoch = float(ramp_up_epoch)
    curr_up_phase = curr_epoch/ramp_up_epoch
    curr_lr = init_lr + (target_lr-init_lr) * (tf.minimum(curr_up_phase, 1))

    # ramp down
    if enable_ramp_down:
        curr_dn_phase = curr_epoch/ramp_down_epoch
        curr_lr *= 0.5 * (tf.cos(np.pi * curr_dn_phase ) + 1)

    return curr_lr


@contextmanager
def name_variable_scope(name_scope_name,
                        var_scope_or_var_scope_name,
                        *var_scope_args,
                        **var_scope_kwargs):
    """A combination of name_scope and variable_scope with different names

    The tf.variable_scope function creates both a name_scope and a variable_scope
    with identical names. But the naming would often be clearer if the names
    of operations didn't inherit the scope name of the (reused) variables.
    So use this function to make shorter and more logical scope names in these cases.
    """
    with tf.name_scope(name_scope_name) as outer_name_scope:
        with tf.variable_scope(var_scope_or_var_scope_name,
                               *var_scope_args,
                               **var_scope_kwargs) as var_scope:
            with tf.name_scope(outer_name_scope) as inner_name_scope:
                yield inner_name_scope, var_scope


@contextmanager
def ema_variable_scope(name_scope_name, var_scope, decay=0.999, zero_debias=False, reuse=None):
    """Scope that replaces trainable variables with their exponential moving averages

    We capture only trainable variables. There's no reason we couldn't support
    other types of variables, but the assumed use case is for trainable variables.
    """
    with tf.name_scope(name_scope_name + "/ema_variables"):
        # Set the variable used for zero debias
        if zero_debias:
            with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                zerodb_var = tf.get_variable(
                        name='zero_debias_num', 
                        shape=[], 
                        initializer=tf.zeros_initializer(), 
                        dtype=tf.int64,
                        trainable=False)
                update_op = tf.assign(zerodb_var, zerodb_var+1)
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op)

        original_trainable_vars = {
            tensor.op.name: tensor
            for tensor
            in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=var_scope.name)
        }
        original_all_vars = {
            tensor.op.name: tensor
            for tensor
            in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=var_scope.name)
        }
        #ema = tf.train.ExponentialMovingAverage(decay)
        #update_op = ema.apply(original_trainable_vars.values())
        new_average_vars = {}
        for each_name, each_tensor in original_trainable_vars.items():
            split_names = each_name.split('/')
            if split_names[1].startswith('__'):
                split_names = split_names[2:]
            new_tensor_name = '/'.join(split_names)
            ave_tensor_name = '%s/DECAY_AVE' % new_tensor_name
            with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                ave_tensor = tf.get_variable(
                    name=ave_tensor_name, 
                    shape=each_tensor.get_shape(), 
                    initializer=tf.zeros_initializer(), 
                    trainable=False)
            new_average_vars[each_name] = ave_tensor
            if not ave_tensor_name in original_all_vars:
                update_op = tf.assign(ave_tensor, ave_tensor * decay + each_tensor * (1 - decay))
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op)

    def use_ema_variables(getter, name, *args, **kwargs):
        #pylint: disable=unused-argument
        if name in original_trainable_vars:
            #return ema.average(original_trainable_vars[name])
            ret_val = new_average_vars[name]
            if zero_debias:
                ret_val = ret_val / (1e-9 + 1 - tf.pow(
                    decay, 
                    tf.cast(zerodb_var+1, tf.float32))/decay)
            return ret_val
        else:
            # For variables not trainable (usually in BNs)
            return getter(name, *args, **kwargs)

    with name_variable_scope(name_scope_name,
                             var_scope,
                             custom_getter=use_ema_variables, reuse=reuse) as (name_scope, var_scope):
        yield name_scope, var_scope
