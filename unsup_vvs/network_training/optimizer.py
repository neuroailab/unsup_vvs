import os
import copy
import tensorflow as tf
import numpy as np
import logging

if 'TFUTILS_LOGFILE' in os.environ:
    logging.basicConfig(filename=os.environ['TFUTILS_LOGFILE'])
    print ("USING LOGFILE: %s" % os.environ['TFUTILS_LOGFILE'])
else:
    logging.basicConfig()

log = logging.getLogger('tfutils')
log.setLevel('DEBUG')


class ClipOptimizerSelf(object):
    """A wrapper for general optimizers.
    This class supports:
    - Clipping the gradients. (controlled by clip parameter)
    - Train part of trainable parameters (controlled by trainable_names)
    Args:
        optimizer_class: Returned value of this function should have `compute_gradients` and `apply_gradients` methods.
        clip (bool, optional): Default is True, clipping by `[-1, 1]`.
    """
    def __init__(
            self, optimizer_class, clip=True, clipping_method='value', clipping_value=1.0, print_global_norm=False,
            trainable_scope=None, *optimizer_args, **optimizer_kwargs):
        self._optimizer = optimizer_class(*optimizer_args, **optimizer_kwargs)
        # The optimizer needs to have these required methods
        required_methods = ['compute_gradients', 'apply_gradients']
        for required_method in required_methods:
            assert required_method in dir(self._optimizer), \
                    "Your optimizer needs to have method %s!" % required_method

        self.clip = clip
        self.clipping_method = clipping_method
        self.clipping_value = clipping_value
        self.print_global_norm = print_global_norm
        self.trainable_scope = trainable_scope

    def compute_gradients(self, loss, var_list=None, *args, **kwargs):
        """Compute gradients to model variables from loss.
        Args:
            loss (tf.Tensor): Tensorflow loss to optimize.
        Returns:
            (tf.Operation): Compute gradient update to model followed by a
            clipping operation if `self.clip` is True.
        """
        # freeze all variables except those with self.trainable_scope in their names
        if var_list is None:
            var_list = tf.trainable_variables()
        if self.trainable_scope is not None:
            new_var_list = [v for v in var_list if any([nm in v.name for nm in self.trainable_scope])]
            if len(new_var_list):
                var_list = new_var_list
                log.info("Only training variables in scope: %s" % self.trainable_scope)
                log.info("variables to be trained: %s" % var_list)

        if var_list is not None:
            num_trainable_params = sum([np.prod(v.shape.as_list()) for v in var_list])
            log.info("Number of Trainable Parameters: %d" % num_trainable_params)

        gvs = self._optimizer.compute_gradients(loss, var_list=var_list,
                                                *args, **kwargs)

        if self.clip:
            if self.clipping_method == "value":
                # gradient clipping. Some gradients returned are 'None' because
                # no relation between the variable and loss; so we skip those.
                gvs = [(tf.clip_by_value(grad, -self.clipping_value, self.clipping_value), var)
                        for grad, var in gvs if grad is not None]
            elif self.clipping_method == "norm":
                print("USING GLOBAL NORM CLIPPING with clip_value %.2f" % self.clipping_value)
                gradients, variables = zip(*gvs)
                norm = tf.global_norm(gradients)
                if self.print_global_norm:
                    norm = tf.Print(norm, [norm], message="grad_global_norm")
                true_fn = lambda: tf.constant(1.0)
                false_fn = lambda: tf.identity(norm)
                norm = tf.case([(tf.logical_or(tf.is_inf(norm), tf.is_nan(norm)), true_fn)], default=false_fn)                
                gradients, global_norm = tf.clip_by_global_norm(gradients, self.clipping_value,
                        use_norm=norm)
                gvs = zip(gradients, variables)
            else:
                raise ValueError("optimizer.clip = True but you didn't specify a valid method in ['value', 'norm']")
        return gvs

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """Apply gradients to model variables specified in `grads_and_vars`.
        `apply_gradients` returns an op that calls
        `tf.train.Optimizer.apply_gradients`
        Args:
            grads_and_vars (list): Description.
            global_step (None, optional): tensorflow global_step variable.
        Returns:
            (tf.Operation): Applies gradient update to model followed by an
                internal gradient zeroing operation to `self.grads_and_vars`.
        """
        optimize = self._optimizer.apply_gradients(grads_and_vars,
                                                   global_step=global_step,
                                                   name=name)
        return optimize
