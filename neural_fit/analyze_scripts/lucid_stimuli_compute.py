# set softtabstop=2 | set shiftwidth=2 | set tabstop=2
import numpy as np
import tensorflow as tf
import sys
import pdb
import os
import argparse
import json
import pickle
import copy
from tqdm import tqdm
from tfutils.imagenet_data import color_normalize

sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./brainscore_mask'))
from cleaned_network_builder import get_network_outputs
from brainscore_mask import tf_model_loader
import bs_fit_neural as bs_fit
from lucid.optvis import objectives, param, transform, render
RESULT_PATH_PATTERN = os.path.join(
        '/mnt/fs4/chengxuz/v4it_temp_results/.result_caching',
        'optimal_stimuli',
        'model_id={model_id}_lucid_raw',
        '{layer}{special}.pkl',
        )
USE_ORG_IMPORT_MODEL = False


def render_vis(model, objective_f, param_f=None, optimizer=None,
               transforms=None, thresholds=(512,), print_objectives=None,
               verbose=False, model_name_scope='encode'):
  """Flexible optimization-based feature vis.

  There's a lot of ways one might wish to customize optimization-based
  feature visualization. It's hard to create an abstraction that stands up
  to all the things one might wish to try.

  This function probably can't do *everything* you want, but it's much more
  flexible than a naive attempt. The basic abstraction is to split the problem
  into several parts. Consider the arguments:

  Args:
    model: The model to be visualized, from Alex' modelzoo.
    objective_f: The objective our visualization maximizes.
      See the objectives module for more details.
    param_f: Paramaterization of the image we're optimizing.
      See the paramaterization module for more details.
      Defaults to a naively paramaterized [1, 128, 128, 3] image.
    optimizer: Optimizer to optimize with. Either tf.train.Optimizer instance,
      or a function from (graph, sess) to such an instance.
      Defaults to Adam with lr .05.
    transforms: A list of stochastic transformations that get composed,
      which our visualization should robustly activate the network against.
      See the transform module for more details.
      Defaults to [transform.jitter(8)].
    thresholds: A list of numbers of optimization steps, at which we should
      save (and display if verbose=True) the visualization.
    print_objectives: A list of objectives separate from those being optimized,
      whose values get logged during the optimization.
    verbose: Should we display the visualization when we hit a threshold?
      This should only be used in IPython.

  Returns:
    2D array of optimization results containing of evaluations of supplied
    param_f snapshotted at specified thresholds. Usually that will mean one or
    multiple channel visualizations stacked on top of each other.
  """

  gpu_options = tf.GPUOptions(allow_growth=True)
  config=tf.ConfigProto(
      allow_soft_placement=True,
      gpu_options=gpu_options,
      )
  with tf.Graph().as_default() as graph, tf.Session(config=config) as sess:

    T = make_vis_T(model, objective_f, param_f, optimizer, transforms)
    loss, vis_op, t_image = T("loss"), T("vis_op"), T("input")
    added_vars = [x for x in tf.global_variables() \
                  if not x.op.name.startswith(model_name_scope)]
    init_new_vars_op = tf.initialize_variables(added_vars)
    init_new_vars_op.run()

    images = []
    all_losses = []
    for i in tqdm(range(max(thresholds)+1)):
      loss_, _ = sess.run([loss, vis_op])
      all_losses.append(loss_)
      if i in thresholds:
        vis = t_image.eval()
        images.append(vis)
        if verbose:
          print(i, loss_)
    return t_image.eval(), all_losses


def make_vis_T(model, objective_f, param_f=None, optimizer=None,
               transforms=None):
  """Even more flexible optimization-base feature vis.

  This function is the inner core of render_vis(), and can be used
  when render_vis() isn't flexible enough. Unfortunately, it's a bit more
  tedious to use:

  >  with tf.Graph().as_default() as graph, tf.Session() as sess:
  >
  >    T = make_vis_T(model, "mixed4a_pre_relu:0")
  >    tf.initialize_all_variables().run()
  >
  >    for i in range(10):
  >      T("vis_op").run()
  >      showarray(T("input").eval()[0])

  This approach allows more control over how the visualizaiton is displayed
  as it renders. It also allows a lot more flexibility in constructing
  objectives / params because the session is already in scope.


  Args:
    model: The model to be visualized, from Alex' modelzoo.
    objective_f: The objective our visualization maximizes.
      See the objectives module for more details.
    param_f: Paramaterization of the image we're optimizing.
      See the paramaterization module for more details.
      Defaults to a naively paramaterized [1, 128, 128, 3] image.
    optimizer: Optimizer to optimize with. Either tf.train.Optimizer instance,
      or a function from (graph, sess) to such an instance.
      Defaults to Adam with lr .05.
    transforms: A list of stochastic transformations that get composed,
      which our visualization should robustly activate the network against.
      See the transform module for more details.
      Defaults to [transform.jitter(8)].

  Returns:
    A function T, which allows access to:
      * T("vis_op") -- the operation for to optimize the visualization
      * T("input") -- the visualization itself
      * T("loss") -- the loss for the visualization
      * T(layer) -- any layer inside the network
  """

  # pylint: disable=unused-variable
  t_image = render.make_t_image(param_f)
  objective_f = objectives.as_objective(objective_f)
  transform_f = render.make_transform_f(transforms)
  optimizer = render.make_optimizer(optimizer, [])

  global_step = tf.train.get_or_create_global_step()
  init_global_step = tf.variables_initializer([global_step])
  init_global_step.run()

  if not USE_ORG_IMPORT_MODEL:
    T = import_model(model, transform_f(t_image), t_image)
  else:
    T = import_model_org(model, transform_f(t_image), t_image)
  loss = objective_f(T)

  vis_op = optimizer.minimize(-loss, global_step=global_step)

  local_vars = locals()
  # pylint: enable=unused-variable

  def T2(name):
    if name in local_vars:
      return local_vars[name]
    else: return T(name)

  return T2


def import_model(model, t_image, t_image_raw):

  ending_points = model(t_image)

  def T(layer):
    if layer == "input": return t_image_raw
    assert layer in ending_points
    return ending_points[layer]

  return T


def import_model_org(model, t_image, t_image_raw):

  model.import_graph(t_image, scope="import", forget_xy_shape=True)

  def T(layer):
    if layer == "input": return t_image_raw
    if layer == "labels": return model.labels
    return t_image.graph.get_tensor_by_name("import/%s:0"%layer)

  return T


def get_lucid_stimuli_compute_parser():
  parser = argparse.ArgumentParser(
      description='Compute the optimal stimuli for raw units using lucid')
  parser.add_argument(
      '--gpu', default='0', type=str, action='store')
  parser = bs_fit.add_load_settings(parser)
  parser = bs_fit.add_model_settings(parser)
  return parser


def get_vm_model_image_losses(args, layer=None):
  model_name_scope = None
  if args.model_type == 'vm_model':
    model_name_scope = 'encode'
  elif args.model_type == 'simclr_model':
    model_name_scope = 'base_model'
  else:
    raise NotImplementedError('Model type %s not supported!' % args.model_type)

  def model(t_image):
    t_image = t_image * 255
    ending_points, _ = get_network_outputs(
        {'images': t_image},
        prep_type=args.prep_type,
        model_type=args.model_type,
        setting_name=args.setting_name,
        module_name=['encode'],
        **json.loads(args.cfg_kwargs))

    all_vars = tf.global_variables()
    var_list = [x for x in all_vars if x.name.startswith(model_name_scope)]
    saver = tf.train.Saver(var_list=var_list)

    if not args.from_scratch:
      if not args.load_from_ckpt:
        model_ckpt_path = tf_model_loader.load_model_from_mgdb(
            db=args.load_dbname,
            col=args.load_colname,
            exp=args.load_expId,
            port=args.load_port,
            cache_dir=args.model_cache_dir,
            step_num=args.load_step,
            )
      else:
        model_ckpt_path = args.load_from_ckpt
      saver.restore(tf.get_default_session(), model_ckpt_path)
    else:
      SESS = tf.get_default_session()
      init_op_global = tf.global_variables_initializer()
      SESS.run(init_op_global)
      init_op_local = tf.local_variables_initializer()
      SESS.run(init_op_local)

    all_train_ref = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
    def _remove_others(vars_ref):
      cp_vars_ref = copy.copy(vars_ref)
      for each_v in cp_vars_ref:
        if each_v.op.name.startswith(model_name_scope):
          vars_ref.remove(each_v)
    _remove_others(all_train_ref)
    return ending_points

  layer = layer or "encode_9"
  batch_size = 16
  param_f = lambda: param.image(224, batch=batch_size)
  num_of_units = 64
  images = []
  all_losses = []
  
  for start_idx in range(0, num_of_units, batch_size):
    obj = objectives.channel(layer, 0 + start_idx, 0)
    for idx in range(1, batch_size):
      obj += objectives.channel(layer, idx + start_idx, idx)
    image, losses = render_vis(model, obj, param_f, model_name_scope=model_name_scope)
    images.append(image)
    all_losses.append(losses)
  images = np.concatenate(images, axis=0)
  all_losses = np.sum(all_losses, axis=0)
  return layer, images, all_losses


def get_lucid_orig_image_losses(args):
  #import lucid.modelzoo.vision_models as models
  #model = models.InceptionV1()
  #model.load_graphdef()
  #param_f = lambda: param.image(128, batch=4)
  #obj = objectives.channel("mixed5a", 9) - 1e2*objectives.diversity("mixed5a")
  #obj = objectives.channel("mixed5a", 9)
  #all_images = render.render_vis(model, obj, param_f)
  #image = all_images[0]

  global USE_ORG_IMPORT_MODEL
  USE_ORG_IMPORT_MODEL = True
  import lucid.modelzoo.vision_models as models
  model = models.InceptionV1()
  model.load_graphdef()
  layer = "mixed5a"
  num_units = 16
  param_f = lambda: param.image(224, batch=num_units)
  obj = objectives.channel(layer, 0, 0)
  for idx in range(1, num_units):
    obj += objectives.channel(layer, idx, idx)
  image, all_losses = render_vis(model, obj, param_f)
  return layer, image, all_losses


def dump_results(args, layer, image, all_losses):
  special = '' 
  if getattr(args, 'identifier', None) is None:
    model_id = '-'.join(
            [args.load_dbname,
             args.load_colname,
             args.load_expId,
             str(args.load_port),
             str(args.load_step)]
            )
  else:
    model_id = args.identifier
  result_path = RESULT_PATH_PATTERN.format(
      model_id = model_id,
      layer = layer,
      special = special,
      )
  save_dir = os.path.dirname(result_path)
  if not os.path.isdir(save_dir):
      os.system('mkdir -p ' + save_dir)
  pickle.dump({'images': image, 'losses': all_losses}, open(result_path, 'wb'))


def main():
  parser = get_lucid_stimuli_compute_parser()
  args = parser.parse_args()
  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
  assert args.set_func, "Must specify set_func"
  args = bs_fit.load_set_func(args)
  layers = bs_fit.TF_RES18_LAYERS
  if getattr(args, 'layers', None) is not None:
      layers = args.layers.split(',')

  for layer in tqdm(layers):
    layer, image, all_losses = get_vm_model_image_losses(args, layer=layer)
    #layer, image, all_losses = get_lucid_orig_image_losses(args)
    dump_results(args, layer, image, all_losses)


if __name__ == '__main__':
  main()
