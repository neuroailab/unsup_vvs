import os, sys
import numpy as np
import tensorflow as tf
import copy
import pdb
import json
import collections

from unsup_vvs.network_training.models.instance_header import get_instance_softmax
import unsup_vvs.network_training.models.rp_col_utils as rc_utils
import unsup_vvs.network_training.models.model_builder as m_builder


def get_ln_data(data_dist, instance_k, instance_data_len, eps=1e-7):
    batch_size = data_dist.get_shape().as_list()[0]
    base_prob = 1.0 / instance_data_len
    ## Pmt
    data_div = data_dist + (instance_k*base_prob + eps)
    ln_data = tf.log(data_dist / data_div)
    ln_data = -tf.reduce_sum(ln_data)/batch_size
    return ln_data


def get_ln_noise(noise_dist, instance_k, instance_data_len, eps=1e-7):
    batch_size = noise_dist.get_shape().as_list()[0]
    base_prob = 1.0 / instance_data_len
    ## Pon
    noise_div = noise_dist + (instance_k*base_prob + eps)
    ln_noise = tf.log((instance_k*base_prob) / noise_div)
    ln_noise = -tf.reduce_sum(ln_noise)/batch_size
    return ln_noise


def instance_loss(
        data_dist, noise_dist, 
        instance_k, instance_data_len
        ):
    if not noise_dist.dtype==tf.int64:
        ln_data = get_ln_data(data_dist, instance_k, instance_data_len)
        ln_noise = get_ln_noise(noise_dist, instance_k, instance_data_len)

        curr_loss = ln_data + ln_noise
        return curr_loss, ln_data, ln_noise
    else:
        return tf.constant(0, dtype=tf.float32),0,0


def sigmoid_rampup(global_step, rampup_length):
    global_step = tf.to_float(global_step)
    rampup_length = tf.to_float(rampup_length)
    def ramp():
        phase = 1.0 - tf.maximum(0.0, global_step) / rampup_length
        return tf.exp(-5.0 * phase * phase)

    result = tf.cond(
            global_step < rampup_length, 
            ramp, 
            lambda: tf.constant(1.0))
    return tf.identity(result, name="sigmoid_rampup")


def get_cons_coefficient(cons_ramp_len, cons_max_value):
    global_step = [v for v in tf.global_variables() if "global_step" in v.name][0]
    sigmoid_rampup_value = sigmoid_rampup(global_step, cons_ramp_len)
    cons_coefficient = tf.multiply(
            tf.constant(cons_max_value, dtype=tf.float32), 
            sigmoid_rampup_value, 
            name='consistency_coefficient')

    return cons_coefficient


def consistency_costs(logits1, logits2, cons_coefficient, name=None):
    with tf.name_scope(name, "consistency_costs") as scope:
        softmax1 = tf.nn.softmax(logits1)
        softmax2 = tf.nn.softmax(logits2)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits1, labels=softmax2)
        entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits2, labels=softmax2)
        costs = cross_entropy - entropy
        costs = costs * cons_coefficient

        mean_cost = tf.reduce_mean(costs, name=scope)
        return mean_cost


def mean_teacher_consitence_and_res(
        class_logit,
        cons_logit,
        ema_class_logit,
        cons_coefficient,
        res_coef,
        ):
    # Stop the gradients here
    ema_class_logit = tf.stop_gradient(ema_class_logit)

    consistence_loss = consistency_costs(
            cons_logit, ema_class_logit, 
            cons_coefficient)
    res_loss = res_coef \
            * tf.nn.l2_loss(class_logit - cons_logit) \
            / np.prod(cons_logit.get_shape().as_list())

    return consistence_loss, res_loss


def is_list_or_tuple(item):
    return isinstance(item, list) or isinstance(item, tuple)


class LossBuilder(object):
    """
    Get losses
    """

    def __init__(
            self,
            cfg_dataset={},
            depth_norm=8000,
            label_norm=20,
            depthloss=0,
            normalloss=0,
            extra_feat=0,
            sm_half_size=0,
            mean_teacher=False,
            res_coef=0.01,
            cons_ramp_len=400000,
            cons_max_value=10.0,
            instance_task=False,
            instance_k=4096,
            instance_data_len=1281025,
            inst_cate_sep=False,
            instance_t=0.07, 
            **kwargs):
        self.cfg_dataset = cfg_dataset
        self.depth_norm = depth_norm
        self.depthloss = depthloss
        self.normalloss = normalloss
        self.extra_feat = extra_feat
        self.sm_half_size = sm_half_size
        self.mean_teacher = mean_teacher
        self.res_coef = res_coef
        self.cons_ramp_len = cons_ramp_len
        self.cons_max_value = cons_max_value
        self.instance_task = instance_task
        self.instance_k = instance_k
        self.instance_t = instance_t
        self.instance_data_len = instance_data_len
        self.inst_cate_sep = inst_cate_sep

    def __get_scenenet_loss(self):
        if self.cfg_dataset.get('scenenet', 0)==1:
            self.loss += self.__get_depth_loss('scenenet')

    def __get_depth_loss(self, which_dataset):
        gt_depth = self.inputs['depth_%s' % which_dataset]
        pred_depth = self.outputs[which_dataset]['depth']
        num_pxls = np.prod(gt_depth.get_shape().as_list())
        loss = tf.nn.l2_loss(pred_depth - gt_depth) / num_pxls
        return loss

    def __get_pbr_loss(self):
        if self.cfg_dataset.get('pbrnet', 0)==1:
            self.loss += self.__get_depth_loss('pbrnet')

    def __get_softmax_loss(self, curr_label, curr_output):
        curr_label = tf.reshape(curr_label, [-1])
        curr_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits = curr_output, labels = curr_label)
        curr_loss = tf.reduce_mean(curr_loss)
        return curr_loss

    def __need_imagenet_inst_clstr(self):
        return self.cfg_dataset.get('imagenet_instance_cluster', 0) == 1

    def __need_prefix_la(self, prefix):
        return self.cfg_dataset.get(
                '{prefix}_LA'.format(prefix=prefix), 0) == 1

    def __need_prefix_ae(self, prefix):
        return self.cfg_dataset.get(
                '{prefix}_ae'.format(prefix=prefix), 0) == 1

    def __need_prefix_cpc(self, prefix):
        return self.cfg_dataset.get(
                '{prefix}_cpc'.format(prefix=prefix), 0) == 1

    def __need_prefix_instance(self, prefix):
        return self.cfg_dataset.get(
                '{prefix}_instance'.format(prefix=prefix), 0) == 1

    def __need_prefix_inst_only_model(self, prefix):
        return self.cfg_dataset.get(
                '{prefix}_instance_only_model'.format(prefix=prefix), 0) == 1

    def __need_prefix_semi_clstr(self, prefix):
        return self.cfg_dataset.get(
                '{prefix}_semi_cluster'.format(prefix=prefix), 0) == 1

    def __need_prefix_mt_clstr(self, prefix):
        return self.cfg_dataset.get(
                '{prefix}_mt_clstr'.format(prefix=prefix), 0) == 1

    def __need_mt_clstr(self):
        return self.__need_prefix_mt_clstr('imagenet') \
                or self.__need_prefix_mt_clstr('imagenet_un')

    def __need_prefix_mt_clstr_model(self, prefix):
        return self.cfg_dataset.get(
                '{prefix}_mt_clstr_model'.format(prefix=prefix), 0) == 1

    def __need_mt_clstr_model(self):
        return self.__need_prefix_mt_clstr_model('imagenet') \
                or self.__need_prefix_mt_clstr_model('imagenet_un')

    def __need_softmax_and_topn(self):
        if 'imagenet_cate' in self.cfg_dataset:
            return self.cfg_dataset.get('imagenet_cate', 1) == 1
        return (not self.__need_imagenet_inst_clstr()) \
                and (not self.instance_task \
                     or self.mean_teacher)

    def __get_inst_model_loss_from_dist(self, dist):
        prob = get_instance_softmax(
                dist, self.instance_t, self.instance_data_len)
        loss = get_ln_data(
                prob, self.instance_k, self.instance_data_len)
        return loss

    def __get_mt_clstr_cons_and_res(
            self,
            which_dataset):
        ema_clstr_output = tf.stop_gradient(
                list(self.outputs['ema'][which_dataset].values())[1])
        clstr_cons = list(self.outputs['primary'][which_dataset].values())[0]
        clstr_output = list(self.outputs['primary'][which_dataset].values())[1]
        ema_clstr_output = tf.nn.l2_normalize(ema_clstr_output, axis=1)
        clstr_cons = tf.nn.l2_normalize(clstr_cons, axis=1)
        clstr_output = tf.nn.l2_normalize(clstr_output, axis=1)

        if not self.__need_mt_clstr_model():
            consistence_loss = self.cons_coefficient \
                    * tf.nn.l2_loss(ema_clstr_output - clstr_cons) \
                    / np.prod(clstr_cons.get_shape().as_list())
            res_loss = self.res_coef \
                    * tf.nn.l2_loss(clstr_output - clstr_cons) \
                    / np.prod(clstr_cons.get_shape().as_list())
        else:
            consistence_dist = tf.reduce_sum(
                    tf.multiply(ema_clstr_output, clstr_cons),
                    axis=1)
            consistence_loss = self.cons_coefficient \
                    * self.__get_inst_model_loss_from_dist(consistence_dist)
            
            res_dist = tf.reduce_sum(
                    tf.multiply(clstr_output, clstr_cons),
                    axis=1)
            res_loss = self.res_coef \
                    * self.__get_inst_model_loss_from_dist(res_dist)

        return consistence_loss, res_loss

    def __get_inst_loss(self, curr_output_dict, name='imagenet'):
        data_prob = curr_output_dict['instance']['data_prob']
        noise_prob = curr_output_dict['instance']['noise_prob']
        if not self.__need_prefix_inst_only_model(name):
            inst_loss, _, _ = instance_loss(
                    data_prob, noise_prob, 
                    self.instance_k, self.instance_data_len)
        else:
            _, inst_loss, _ = instance_loss(
                    data_prob, noise_prob, 
                    self.instance_k, self.instance_data_len)
        return inst_loss

    def __get_l1_loss(self, l1_weight, activation):
        curr_loss = l1_weight * tf.reduce_sum(tf.abs(activation)) \
                / np.prod(activation.get_shape().as_list())
        return curr_loss

    def __get_ae_reconstruct_loss(self, curr_output_dict):
        normalized_image = tf.cast(self.inputs['image_imagenet'], tf.float32)
        normalized_image = m_builder.color_normalize(normalized_image / 255)
        diff_image = curr_output_dict['ae_output'] - normalized_image
        curr_loss = tf.nn.l2_loss(diff_image) \
                / np.prod(diff_image.get_shape().as_list())
        return curr_loss

    def __get_ae_l1_loss(self, single=True):
        l1_weight = self.cfg_dataset.get('imagenet_ae_l1', 0)
        if l1_weight > 0:
            all_ae_outputs = tf.get_collection('AE_embedding')
            if single:
                l1_loss = self.__get_l1_loss(l1_weight, all_ae_outputs[-1])
            else:
                all_ae_outputs = tf.concat(all_ae_outputs, axis=0)
                #print_op = tf.print(
                #        'AE embd, ', 
                #        tf.reduce_sum(tf.abs(all_ae_outputs)))
                print_op = tf.no_op()
                with tf.control_dependencies([print_op]):
                    l1_loss = self.__get_l1_loss(l1_weight, all_ae_outputs)
        else:
            l1_loss = 0
        return l1_loss

    def __get_ae_loss(self, curr_output_dict):
        curr_loss = self.__get_ae_reconstruct_loss(curr_output_dict)
        curr_loss += self.__get_ae_l1_loss()
        return curr_loss

    def __get_imagenet_loss(self):
        if self.cfg_dataset.get('imagenet', 0)==1 \
                or self.cfg_dataset.get('rp', 0)==1 \
                or self.cfg_dataset.get('colorization', 0)==1:
            curr_loss = 0
            if self.mean_teacher or self.__need_mt_clstr():
                curr_output_dict = self.outputs['primary']['imagenet']
            else:
                curr_output_dict = self.outputs['imagenet']

            if self.__need_softmax_and_topn():
                curr_output = list(curr_output_dict.values())[0]
                if not isinstance(curr_output, list):
                    curr_output = [curr_output]
                for each_curr_output in curr_output:
                    curr_loss += self.__get_softmax_loss(
                            curr_label=self.inputs['label_imagenet'],
                            curr_output=each_curr_output)

            if (self.instance_task or self.__need_prefix_instance('imagenet')) \
                    and not self.mean_teacher:
                inst_loss = self.__get_inst_loss(curr_output_dict)
                curr_loss += inst_loss

            if self.__need_imagenet_inst_clstr():
                curr_loss += curr_output_dict['inst_clstr']['loss']

            if self.__need_prefix_la('imagenet'):
                curr_loss += curr_output_dict['LA']['loss']

            if self.mean_teacher:
                ema_class_logit = list(self.outputs['ema']['imagenet'].values())[0]
                consistence_loss, res_loss = mean_teacher_consitence_and_res(
                        class_logit=list(curr_output_dict.values())[0],
                        cons_logit=list(curr_output_dict.values())[1],
                        ema_class_logit=ema_class_logit,
                        cons_coefficient=self.cons_coefficient,
                        res_coef=self.res_coef)
                curr_loss += consistence_loss + res_loss

            if self.__need_mt_clstr():
                consistence_loss, res_loss = self.__get_mt_clstr_cons_and_res(
                        'imagenet')
                curr_loss += consistence_loss + res_loss

            if self.__need_prefix_ae('imagenet'):
                curr_loss += self.__get_ae_loss(curr_output_dict)

            if self.__need_prefix_cpc('imagenet'):
                curr_loss += curr_output_dict['cpc_loss']

            self.loss += curr_loss

    def __get_imagenet_branch2_loss(self):
        name = 'imagenet_branch2'
        if self.cfg_dataset.get(name, 0)==1:
            curr_loss = 0
            curr_output_dict = self.outputs[name]

            if self.__need_prefix_instance(name):
                inst_loss = self.__get_inst_loss(
                        curr_output_dict, name)
                curr_loss += inst_loss

            if self.__need_prefix_la(name):
                curr_loss += curr_output_dict['LA']['loss']
            self.loss += curr_loss

    def __get_imagenet_un_loss(self):
        if self.cfg_dataset.get('imagenet_un', 0)==1:
            curr_loss = 0
            if self.mean_teacher or self.__need_mt_clstr():
                curr_output_dict = self.outputs['primary']['imagenet_un']
            else:
                curr_output_dict = self.outputs['imagenet_un']

            if self.instance_task or self.__need_prefix_instance('imagenet_un'):
                inst_loss = self.__get_inst_loss(
                        curr_output_dict, 'imagenet_un')
                curr_loss += inst_loss

            if self.__need_prefix_semi_clstr('imagenet_un'):
                curr_loss += curr_output_dict['semi_clstr']['loss']

            if self.mean_teacher:
                ema_class_logit = list(self.outputs['ema']['imagenet_un'].values())[0]
                consistence_loss, res_loss = mean_teacher_consitence_and_res(
                        class_logit=list(curr_output_dict.values())[0],
                        cons_logit=list(curr_output_dict.values())[1],
                        ema_class_logit=ema_class_logit,
                        cons_coefficient=self.cons_coefficient,
                        res_coef=self.res_coef)
                curr_loss += consistence_loss + res_loss # Add all three losses

            if self.__need_mt_clstr():
                consistence_loss, res_loss = self.__get_mt_clstr_cons_and_res(
                        'imagenet_un')
                curr_loss += consistence_loss + res_loss

            self.loss += curr_loss

    def __get_rp_loss(self, sub_dataset_name):
        curr_dataset_name = 'rp_%s' % sub_dataset_name
        if self.cfg_dataset.get(curr_dataset_name, 0)==1:
            curr_output_dict = self.outputs[curr_dataset_name]
            rp_loss = rc_utils.get_rp_loss(curr_output_dict['rp_category'])
            self.loss += rp_loss

    def __get_col_loss(self, sub_dataset_name):
        curr_dataset_name = 'col_%s' % sub_dataset_name
        if self.cfg_dataset.get(curr_dataset_name, 0)==1:
            col_loss = rc_utils.get_col_loss(
                    self.outputs[curr_dataset_name]['colorization_head'],
                    self.inputs['q_label_col_%s' % sub_dataset_name])
            self.loss += col_loss

    def __get_saycam_loss(self):
        if self.cfg_dataset.get('saycam', 0)==1:
            curr_output_dict = self.outputs['saycam']
            if (self.instance_task or self.__need_prefix_instance('saycam')):
                inst_loss = self.__get_inst_loss(curr_output_dict)
                self.loss += inst_loss

            if self.__need_prefix_la('saycam'):
                self.loss += curr_output_dict['LA']['loss']

    def __get_coco_loss(self):
        if self.cfg_dataset.get('coco', 0)==1:
            raise NotImplementedError("COCO")

    def __get_place_loss(self):
        if self.cfg_dataset.get('place', 0)==1:
            raise NotImplementedError("Place")

    def __get_kinetics_loss(self):
        if self.cfg_dataset.get('kinetics', 0)==1:
            raise NotImplementedError("Kinetics")

    def __get_nyu_loss(self):
        if self.cfg_dataset.get('nyuv2', 0)==1:
            raise NotImplementedError("Nyuv2")

    def __set_cons_coef(self):
        self.cons_coefficient = get_cons_coefficient(
                self.cons_ramp_len, self.cons_max_value)

    def get_loss(self, outputs, inputs, **kwargs):
        self.outputs = outputs
        self.inputs = inputs

        self.__set_cons_coef()

        self.loss = 0
        self.__get_scenenet_loss()
        self.__get_pbr_loss()
        self.__get_imagenet_loss()
        self.__get_imagenet_branch2_loss()
        self.__get_imagenet_un_loss()
        self.__get_rp_loss('imagenet')
        self.__get_col_loss('imagenet')
        self.__get_coco_loss()
        self.__get_place_loss()
        self.__get_kinetics_loss()
        self.__get_nyu_loss()
        self.__get_saycam_loss()

        ret_loss = self.loss
        self.outputs = None
        self.inputs = None
        self.loss = None
        self.cons_coefficient = None
        return ret_loss

    def __get_scenenet_metrics(self):
        if self.cfg_dataset.get('scenenet', 0)==1:
            self.loss_dict['scenenet_depth'] = self.__get_depth_loss('scenenet')

    def __get_pbr_metrics(self):
        if self.cfg_dataset.get('pbrnet', 0)==1:
            self.loss_dict['pbrnet_depth'] = self.__get_depth_loss('pbrnet')

    def __get_coco_metrics(self):
        if self.cfg_dataset.get('coco', 0)==1:
            raise NotImplementedError("COCO")

    def __get_place_metrics(self):
        if self.cfg_dataset.get('place', 0)==1:
            raise NotImplementedError("Place")

    def __get_kinetics_metrics(self):
        if self.cfg_dataset.get('kinetics', 0)==1:
            raise NotImplementedError("Kinetics")

    def __get_nyu_metrics(self):
        if self.cfg_dataset.get('nyuv2', 0)==1:
            raise NotImplementedError("Nyuv2")

    def __add_topn_report(
            self, curr_label, curr_output, 
            str_suffix = 'imagenet'):
        curr_label = tf.reshape(curr_label, [-1])
        if not isinstance(curr_output, list):
            curr_top1 = tf.nn.in_top_k(curr_output, curr_label, 1)
            curr_top5 = tf.nn.in_top_k(curr_output, curr_label, 5)
            self.loss_dict['loss_top1_%s' % str_suffix] = curr_top1
            self.loss_dict['loss_top5_%s' % str_suffix] = curr_top5
        else:
            for idx, each_out in enumerate(curr_output):
                curr_top1 = tf.nn.in_top_k(each_out, curr_label, 1)
                curr_top5 = tf.nn.in_top_k(each_out, curr_label, 5)
                self.loss_dict['loss_top1_%i_%s' % (idx, str_suffix)] = curr_top1
                self.loss_dict['loss_top5_%i_%s' % (idx, str_suffix)] = curr_top5

    def __get_curr_top1_from_inst(self, curr_dist, all_labels, temp_truth):
        if is_list_or_tuple(all_labels):
            all_labels = all_labels[0]

        _, top_indices = tf.nn.top_k(curr_dist, k=1)
        curr_pred = tf.gather(
                all_labels, 
                tf.squeeze(top_indices, axis=1))
        curr_top1 = tf.reduce_mean(
                tf.cast(
                    tf.equal(curr_pred, tf.cast(temp_truth, tf.int64)), 
                    tf.float32))
        return curr_top1

    def __get_imagenet_metrics(self):
        if self.cfg_dataset.get('imagenet', 0)==1:
            if self.mean_teacher or self.__need_mt_clstr():
                curr_output_dict = self.outputs['primary']['imagenet']
            else:
                curr_output_dict = self.outputs['imagenet']
            curr_truth = self.inputs['label_imagenet']

            if self.__need_softmax_and_topn():
                curr_output = list(curr_output_dict.values())[0]
                self.__add_topn_report(
                        curr_label=curr_truth,
                        curr_output=curr_output)

            if self.__need_imagenet_inst_clstr():
                curr_dist = curr_output_dict['inst_clstr'][0]
                all_labels = curr_output_dict['inst_clstr'][1]
                self.loss_dict['imagenet_top1_clstr'] \
                        = self.__get_curr_top1_from_inst(
                                curr_dist, all_labels, curr_truth)

            if self.instance_task or self.__need_prefix_instance('imagenet'):
                curr_dist = curr_output_dict['instance'][0]
                all_labels = curr_output_dict['instance'][1]
                self.loss_dict['imagenet_top1'] \
                        = self.__get_curr_top1_from_inst(
                                curr_dist, all_labels, curr_truth)

            if self.__need_prefix_la('imagenet'):
                curr_dist = curr_output_dict['LA'][0]
                all_labels = curr_output_dict['LA'][1]
                self.loss_dict['la_imagenet_top1'] \
                        = self.__get_curr_top1_from_inst(
                                curr_dist, all_labels, curr_truth)

            if self.mean_teacher:
                curr_output = list(self.outputs['ema']['imagenet'].values())[0]
                self.__add_topn_report(
                        curr_label=curr_truth,
                        curr_output=curr_output,
                        str_suffix='imagenet_ema')

            if self.__need_mt_clstr():
                curr_dist = self.outputs['ema']['imagenet']['inst_clstr'][0]
                all_labels = self.outputs['ema']['imagenet']['inst_clstr'][1]
                self.loss_dict['ema_imagenet_top1_clstr'] \
                        = self.__get_curr_top1_from_inst(
                                curr_dist, all_labels, curr_truth)

            if self.__need_prefix_ae('imagenet'):
                self.loss_dict['imagenet_ae_loss'] = \
                        self.__get_ae_loss(curr_output_dict)

            if self.__need_prefix_cpc('imagenet'):
                self.loss_dict['imagenet_cpc_loss'] = \
                        curr_output_dict['cpc_loss']

    def __get_imagenet_branch2_metrics(self):
        name = 'imagenet_branch2'
        if self.cfg_dataset.get(name, 0)==1:
            curr_output_dict = self.outputs[name]
            curr_truth = self.inputs['label_' + name]

            if self.instance_task or self.__need_prefix_instance(name):
                curr_dist = curr_output_dict['instance'][0]
                all_labels = curr_output_dict['instance'][1]
                self.loss_dict[name + '_top1'] \
                        = self.__get_curr_top1_from_inst(
                                curr_dist, all_labels, curr_truth)

            if self.__need_prefix_la(name):
                curr_dist = curr_output_dict['LA'][0]
                all_labels = curr_output_dict['LA'][1]
                self.loss_dict['la_%s_top1' % name] \
                        = self.__get_curr_top1_from_inst(
                                curr_dist, all_labels, curr_truth)

    def __get_imagenet_un_metrics(self):
        if self.mean_teacher or self.__need_mt_clstr():
            return 

        if self.cfg_dataset.get('imagenet_un', 0)==1:
            curr_output_dict = self.outputs['imagenet_un']
            curr_truth = self.inputs['label_imagenet_un']

            if self.__need_prefix_instance('imagenet_un'):
                curr_dist = curr_output_dict['instance'][0]
                all_labels = curr_output_dict['instance'][1]

                assert 'imagenet_top1' not in self.loss_dict, "Replicate key!"
                self.loss_dict['imagenet_top1'] \
                        = self.__get_curr_top1_from_inst(
                                curr_dist, all_labels, curr_truth)

            if self.__need_prefix_semi_clstr('imagenet_un'):
                curr_dist = curr_output_dict['semi_clstr'][0]
                all_labels = curr_output_dict['semi_clstr'][1]
                self.loss_dict['imagenet_top1_semi'] \
                        = self.__get_curr_top1_from_inst(
                                curr_dist, all_labels, curr_truth)

    def __get_rp_metrics(self, sub_dataset_name):
        curr_dataset_name = 'rp_%s' % sub_dataset_name
        if self.cfg_dataset.get(curr_dataset_name, 0)==1:
            curr_output_dict = self.outputs[curr_dataset_name]
            rp_top1 = rc_utils.get_rp_top1(curr_output_dict['rp_category'])
            self.loss_dict['rp_%s_top1' % sub_dataset_name] = rp_top1

    def __get_col_metrics(self, sub_dataset_name):
        curr_dataset_name = 'col_%s' % sub_dataset_name
        if self.cfg_dataset.get(curr_dataset_name, 0)==1:
            col_top1 = rc_utils.get_col_top1(
                    self.outputs[curr_dataset_name]['colorization_head'],
                    self.inputs['q_label_col_%s' % sub_dataset_name])
            self.loss_dict['col_%s_top1' % sub_dataset_name] = col_top1

    def get_val_metrics(self, inputs, output, cfg_dataset, **kwargs):
        self.cfg_dataset = cfg_dataset

        self.outputs = output
        self.inputs = inputs
        self.loss_dict = {}

        self.__get_scenenet_metrics()
        self.__get_pbr_metrics()
        self.__get_imagenet_metrics()
        self.__get_imagenet_branch2_metrics()
        self.__get_imagenet_un_metrics()
        self.__get_rp_metrics('imagenet')
        self.__get_col_metrics('imagenet')
        self.__get_coco_metrics()
        self.__get_place_metrics()
        self.__get_kinetics_metrics()
        self.__get_nyu_metrics()

        loss_dict = self.loss_dict
        self.outputs = None
        self.inputs = None
        self.loss_dict = None
        return loss_dict

    def __need_rep_softmax_loss(self):
        if 'imagenet_cate' in self.cfg_dataset:
            return self.cfg_dataset.get('imagenet_cate', 1) == 1
        return self.mean_teacher and (not self.__need_imagenet_inst_clstr())

    def __get_rep_softmax_loss(self):
        if self.__need_rep_softmax_loss():
            if self.mean_teacher:
                curr_output = list(self.outputs['primary']['imagenet'].values())[0]
            else:
                curr_output = list(self.outputs['imagenet'].values())[0]

            cate_loss = self.__get_softmax_loss(
                    curr_label=self.inputs['label_imagenet'], 
                    curr_output=curr_output)
            self.ret_dict['loss_cate'] = cate_loss

    def __get_instance_update_ops(
            self, 
            all_label_list, memory_bank_list,
            data_indx, data_label, new_memory):
        if not is_list_or_tuple(memory_bank_list):
            memory_bank_list = [memory_bank_list]
        if not is_list_or_tuple(all_label_list):
            all_label_list = [all_label_list]

        all_update_ops = []
        for gpu_device, all_label, memory_bank \
                in zip(self.devices, all_label_list, memory_bank_list):
            with tf.device(gpu_device):
                ## Update label and memory vector here
                lb_update_op = tf.scatter_update(
                        all_label, data_indx, 
                        tf.cast(data_label, tf.int64))
                mb_update_op = tf.scatter_update(
                        memory_bank, data_indx, 
                        new_memory)

                all_update_ops.append(lb_update_op)
                all_update_ops.append(mb_update_op)
        return all_update_ops

    def __get_semi_clstr_update_ops(
            self, data_indx, 
            new_clstr_labels, clstr_labels):
        if not is_list_or_tuple(clstr_labels):
            clstr_labels = [clstr_labels]
        all_update_ops = []
        for gpu_device, clstr_label \
                in zip(self.devices, clstr_labels):
            with tf.device(gpu_device):
                ## Update label and memory vector here
                clstr_lb_update_op = tf.scatter_update(
                        clstr_label, data_indx, 
                        tf.cast(new_clstr_labels, tf.int64))

                all_update_ops.append(clstr_lb_update_op)
        return all_update_ops

    def __get_inst_loss_with_updates(
            self,
            data_dist,
            noise_dist,
            memory_bank_list,
            all_label_list,
            data_indx,
            new_memory,
            data_label):
        all_update_ops = self.__get_instance_update_ops(
                all_label_list, memory_bank_list,
                data_indx, data_label, new_memory)
        with tf.control_dependencies(all_update_ops):
            _, loss_model, loss_noise = instance_loss(
                    data_dist, noise_dist, 
                    self.instance_k, self.instance_data_len
                    )
        return loss_model, loss_noise

    def __get_rep_instance_loss(self):
        if self.instance_task \
                or self.__need_prefix_instance('imagenet') \
                or self.__need_prefix_instance('imagenet_un') \
                or self.__need_prefix_instance('imagenet_branch2') \
                or self.__need_prefix_instance('saycam'):
            curr_out_dict = None
            if (not self.inst_cate_sep and not self.mean_teacher) \
               and self.__need_prefix_instance('imagenet'):
                curr_out_dict = self.outputs['imagenet']
                data_label = self.inputs['label_imagenet']
            if self.inst_cate_sep or self.__need_prefix_instance('imagenet_un'):
                curr_out_dict = self.outputs['imagenet_un']
                data_label = self.inputs['label_imagenet_un']
            if self.__need_prefix_instance('imagenet_branch2'):
                curr_out_dict = self.outputs['imagenet_branch2']
                data_label = self.inputs['label_imagenet_branch2']
            if self.mean_teacher:
                curr_out_dict = self.outputs['primary']['imagenet_un']
                data_label = self.inputs['label_imagenet_un']
            if self.__need_prefix_instance('saycam'):
                curr_out_dict = self.outputs['saycam']
                data_label = self.inputs['label_saycam']
            if (curr_out_dict is None) and ('imagenet' in self.outputs): 
                curr_out_dict = self.outputs['imagenet']
                data_label = self.inputs['label_imagenet']
            curr_out_dict = curr_out_dict['instance']

            loss_model, loss_noise = self.__get_inst_loss_with_updates(
                    data_dist=curr_out_dict['data_prob'],
                    noise_dist=curr_out_dict['noise_prob'],
                    memory_bank_list=curr_out_dict['memory_bank'],
                    all_label_list=curr_out_dict['all_labels'],
                    data_indx=curr_out_dict['data_indx'],
                    new_memory=curr_out_dict['new_data_memory'],
                    data_label=data_label)
            if self.__need_prefix_inst_only_model('imagenet') \
                    or self.__need_prefix_inst_only_model('imagenet_un'):
                self.ret_dict.update({
                        'loss_model': loss_model})
            else:
                self.ret_dict.update({
                        'loss_model': loss_model,
                        'loss_noise': loss_noise})

    def __get_rep_la_loss(self):
        if self.__need_prefix_la('imagenet') or self.__need_prefix_la('saycam'):
            if self.__need_prefix_la('imagenet'):
                curr_out_dict = self.outputs['imagenet']['LA']
                data_label=self.inputs['label_imagenet']
            if self.__need_prefix_la('saycam'):
                curr_out_dict = self.outputs['saycam']['LA']
                data_label=self.inputs['label_saycam']

            all_update_ops = self.__get_instance_update_ops(
                    all_label_list=curr_out_dict['all_labels'], 
                    memory_bank_list=curr_out_dict['memory_bank'],
                    data_indx=curr_out_dict['data_indx'], 
                    data_label=data_label, 
                    new_memory=curr_out_dict['new_data_memory'])

            with tf.control_dependencies(all_update_ops):
                loss_la = tf.identity(curr_out_dict['loss'])
            self.ret_dict['loss_la'] = loss_la

    def __get_rep_inst_clstr_loss(self):
        if self.__need_imagenet_inst_clstr():
            if not self.__need_mt_clstr():
                curr_out_dict = self.outputs['imagenet']['inst_clstr']
            else:
                curr_out_dict = self.outputs['primary']['imagenet']['inst_clstr']

            data_label = self.inputs['label_imagenet']
            data_indx = self.inputs['index_imagenet']

            all_update_ops = self.__get_instance_update_ops(
                    memory_bank_list=curr_out_dict['memory_bank'],
                    all_label_list=curr_out_dict['all_labels'],
                    new_memory=curr_out_dict['new_data_memory'],
                    data_indx=data_indx, 
                    data_label=data_label)
            with tf.control_dependencies(all_update_ops):
                loss_pure = tf.identity(curr_out_dict['loss'])
            self.ret_dict['loss_clstr'] = loss_pure

    def __get_rep_semi_clstr_loss(self):
        if self.__need_prefix_semi_clstr('imagenet_un'):
            curr_out_dict = self.outputs['imagenet_un']['semi_clstr']
            data_label = self.inputs['label_imagenet']
            data_indx = self.inputs['index_imagenet']

            all_update_ops = self.__get_instance_update_ops(
                    memory_bank_list=curr_out_dict['memory_bank'],
                    all_label_list=curr_out_dict['all_labels'],
                    new_memory=curr_out_dict['new_data_memory'],
                    data_indx=data_indx, 
                    data_label=data_label)

            clstr_lbl_update_ops = self.__get_semi_clstr_update_ops(
                    data_indx=data_indx,
                    new_clstr_labels=curr_out_dict['new_clstr_labels'],
                    clstr_labels=curr_out_dict['clstr_labels'])
            all_update_ops.extend(clstr_lbl_update_ops)

            with tf.control_dependencies(all_update_ops):
                loss_pure = tf.identity(curr_out_dict['loss'])
            self.ret_dict['loss_semi_clstr'] = loss_pure

    def __get_rep_mean_teacher_loss(self):
        if self.mean_teacher:
            curr_out_dict = self.outputs['primary']['imagenet']
            ema_class_logit = list(self.outputs['ema']['imagenet'].values())[0]
            consistence_loss_0, res_loss_0 = mean_teacher_consitence_and_res(
                    class_logit=list(curr_output_dict.values())[0],
                    cons_logit=list(curr_output_dict.values())[1],
                    ema_class_logit=ema_class_logit,
                    cons_coefficient=self.cons_coefficient,
                    res_coef=self.res_coef)

            curr_out_dict = self.outputs['primary']['imagenet_un']
            ema_class_logit = list(self.outputs['ema']['imagenet_un'].values())[0]
            consistence_loss, res_loss = mean_teacher_consitence_and_res(
                    class_logit=list(curr_output_dict.values())[0],
                    cons_logit=list(curr_output_dict.values())[1],
                    ema_class_logit=ema_class_logit,
                    cons_coefficient=self.cons_coefficient,
                    res_coef=self.res_coef)
            self.ret_dict.update({
                    'mt_con': (consistence_loss_0 + consistence_loss_1)/2,
                    'mt_res': (res_loss_0 + res_loss_1)/2,
                    })

    def __get_rep_mt_clstr_loss(self):
        if self.__need_mt_clstr():
            consistence_loss_0, res_loss_0 = self.__get_mt_clstr_cons_and_res(
                    'imagenet')
            consistence_loss_1, res_loss_1 = self.__get_mt_clstr_cons_and_res(
                    'imagenet_un')
            self.ret_dict.update({
                    'mt_con': (consistence_loss_0 + consistence_loss_1)/2,
                    'mt_res': (res_loss_0 + res_loss_1)/2,
                    })

    def __get_rep_rp(self, sub_dataset_name):
        curr_dataset_name = 'rp_%s' % sub_dataset_name
        if self.cfg_dataset.get(curr_dataset_name, 0)==1:
            curr_output_dict = self.outputs[curr_dataset_name]
            rp_top1 = rc_utils.get_rp_top1(curr_output_dict['rp_category'])
            self.ret_dict['rp_%s_top1' % sub_dataset_name] = rp_top1

    def __get_rep_col(self, sub_dataset_name):
        curr_dataset_name = 'col_%s' % sub_dataset_name
        if self.cfg_dataset.get(curr_dataset_name, 0)==1:
            col_top1 = rc_utils.get_col_top1(
                    self.outputs[curr_dataset_name]['colorization_head'],
                    self.inputs['q_label_col_%s' % sub_dataset_name])
            self.ret_dict['col_%s_top1' % sub_dataset_name] = col_top1

    def __get_rep_ae_loss(self):
        if self.__need_prefix_ae('imagenet'):
            re_loss = self.__get_ae_reconstruct_loss(self.outputs['imagenet'])
            self.ret_dict['ae_re'] = re_loss
            l1_loss = self.__get_ae_l1_loss(single=False)
            if l1_loss is not 0:
                self.ret_dict['ae_l1'] = l1_loss

    def get_rep_losses_and_updates(self, inputs, output, devices):
        self.outputs = output
        self.inputs = inputs
        self.devices = devices

        self.__set_cons_coef()
        self.ret_dict = {}

        self.__get_rep_softmax_loss()
        self.__get_rep_instance_loss()
        self.__get_rep_la_loss()
        self.__get_rep_inst_clstr_loss()
        self.__get_rep_semi_clstr_loss()
        self.__get_rep_mean_teacher_loss()
        self.__get_rep_mt_clstr_loss()
        self.__get_rep_rp('imagenet')
        self.__get_rep_col('imagenet')
        self.__get_rep_ae_loss()

        ret_dict = self.ret_dict
        self.outputs = None
        self.inputs = None
        self.cons_coefficient = None
        self.ret_dict = None
        return ret_dict

    def __get_imagenet_feats(self):
        if self.cfg_dataset.get('imagenet', 0)==1:
            curr_output_dict = self.outputs['imagenet']
            self.ret_dict['fea_image_imagenet'] = self.inputs['image_imagenet']

            if self.__need_prefix_ae('imagenet'):
                out_image = curr_output_dict['ae_output']
                out_image = m_builder.color_denormalize(out_image) * 255
                self.ret_dict['out_image_imagenet'] \
                        = tf.cast(out_image, tf.uint8)

    def get_feat_targets(self, inputs, output, num_to_save):
        self.outputs = output
        self.inputs = inputs
        self.ret_dict = {}

        self.__get_imagenet_feats()

        ret_dict = self.ret_dict
        self.outputs = None
        self.inputs = None
        self.ret_dict = None
        return ret_dict

    def get_pca_targets(self, inputs, output):
        ret_dict = output
        ret_dict['_inputs'] = inputs
        return ret_dict
