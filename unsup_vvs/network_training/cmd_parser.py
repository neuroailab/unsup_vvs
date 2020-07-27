import argparse
from unsup_vvs.network_training.exp_settings.utils import get_setting_func


def add_general_setting(parser):
    # General settings
    parser.add_argument(
	    '--nport', default=27017, type=int, action='store',
            help='Port number of mongodb')
    parser.add_argument(
	    '--dbname', default="combinet-test", type=str, action='store',
            help='Database name')
    parser.add_argument(
	    '--collname', default="combinet", type=str, action='store',
            help='Collection name')
    parser.add_argument(
	    '--expId', default="combinet_test", type=str, action='store',
            help='Name of experiment id')
    parser.add_argument(
	    '--cacheDirPrefix', default="/mnt/fs1/chengxuz", 
            type=str, action='store',
            help='Prefix of cache directory')
    parser.add_argument(
	    '--innerargs', default=[], type=str, action='append',
            help='Arguments for every network')
    parser.add_argument(
	    '--with_rep', default=0, type=int, action='store',
            help='Whether reporting other losses every batch')
    parser.add_argument(
	    '--with_train', default=0, type=int, action='store',
            help='Whether with training dataset')
    parser.add_argument(
	    '--valid_first', default=0, type=int, action='store',
            help='Whether validating first')
    parser.add_argument(
	    '--with_feat', default=1, type=int, action='store',
            help='Whether adding the feat validation')
    parser.add_argument(
	    '--col_image', default=0, type=int, action='store',
            help='Whether output colorization images')
    parser.add_argument(
	    '--sm_gen_features', default=0, type=int, action='store', # sm_add
            help='Whether generate features from layers')
    parser.add_argument(
	    '--sm_train_or_test', default=0, type=int, action='store', # sm_add
            help='Whether train or test the model')
    parser.add_argument(
	    '--sm_loaddir', 
            default="gs://full_size_imagenet/image_label_full/", 
            type=str, action='store',
            help='This is the tpu_dataset_path')
    parser.add_argument(
	    '--sm_loaddir2', default="gs://sm-dataset/", 
            type=str, action='store',
            help='This is the second tpu_dataset_path,'
                    + 'because some tasks need two dataset')
    parser.add_argument(
	    '--sm_loaddir3', default="gs://sm-dataset/", 
            type=str, action='store',
            help='This is the third tpu_dataset_path, '\
                    + 'because some tasks need two dataset')
    parser.add_argument(
	    '--val_on_train', default=0, type=int, action='store',
            help='Whether validating on train tfrecords')
    parser.add_argument(
	    '--load_setting_func', default=None, type=str, action='store',
            help='Saved setting function')

    ## Load metric
    parser.add_argument(
	    '--ckpt_file', default=None, type=str, action='store',
            help='Whether load the model from ckpt file')
    parser.add_argument(
	    '--loadport', default=None, type=int, action='store',
            help='Port number of mongodb for loading')
    parser.add_argument(
	    '--loadstep', default=None, type=int, action='store',
            help='Number of steps for loading')
    parser.add_argument(
	    '--load_dbname', default=None, type=str, action='store',
            help='Load database name')
    parser.add_argument(
	    '--load_collname', default=None, type=str, action='store',
            help='Load collection name')
    parser.add_argument(
	    '--loadexpId', default=None, type=str, action='store',
            help='Name of experiment id')
    parser.add_argument(
	    '--load_param_dict', default=None, type=str, action='store',
            help='Load param dict used in tfutils load_params')
    parser.add_argument(
	    '--drop_global_step', action='store_true',
            help='Whether dropping the global step')

    ## Saving metric
    parser.add_argument(
	    '--fre_valid', default=10000, type=int, action='store',
            help='Frequency of the validation')
    parser.add_argument(
	    '--fre_metric', default=1000, type=int, action='store',
            help='Frequency of saving metrics')
    parser.add_argument(
	    '--fre_filter', default=10000, type=int, action='store',
            help='Frequency of saving filters')
    parser.add_argument(
	    '--fre_cache_filter', default=None, type=int, action='store',
            help='Frequency of cahcing filters')
    parser.add_argument(
	    '--checkpoint_max', default=300, type=int, action='store',
            help='TPU checkpoint saved number')
    return parser


def add_network_setting(parser):
    # Network related
    parser.add_argument(
	    '--pathconfig', default="normals_config_fcnvgg16_withdepth.cfg", 
            type=str, action='store',
            help='Path to config file')
    parser.add_argument(
	    '--network_func', default=None, 
            type=str, action='store',
            help='Function name stored in network_cfg_funcs')
    parser.add_argument(
	    '--network_func_kwargs', default=None, 
            type=str, action='store',
            help='Kwargs for function stored in network_cfg_funcs')
    parser.add_argument(
	    '--dataconfig', default="dataset_config.cfg", 
            type=str, action='store',
            help='Path to config file for dataset')
    parser.add_argument(
	    '--valdconfig', default=None, type=str, action='store',
            help='Validation dataset config, default to be None, \
                    and will copy to other configs below')
    parser.add_argument(
	    '--topndconfig', default=None, type=str, action='store',
            help='Path to config file for dataset, for topn validation')
    parser.add_argument(
	    '--featdconfig', default=None, type=str, action='store',
            help='Path to config file for dataset, for feats validation')
    parser.add_argument(
	    '--modeldconfig', default=None, type=str, action='store',
            help='Path to config file for dataset, for model in validation')
    parser.add_argument(
	    '--seed', default=0, type=int, action='store',
            help='Random seed for model')
    parser.add_argument(
	    '--namefunc', default="combine_tfutils_general", 
            type=str, action='store',
            help='Name of function to build the network')
    parser.add_argument(
	    '--valinum', default=-1, type=int, action='store',
            help='Number of validation steps, default is -1, \
                    which means all the validation')
    parser.add_argument(
	    '--cache_filter', default=0, type=int, action='store',
            help='Whether cache the pretrained weights as tf tensors')
    parser.add_argument(
	    '--fix_pretrain', default=0, type=int, action='store',
            help='Whether fix the pretrained weights')
    parser.add_argument(
	    '--extra_feat', default=0, type=int, action='store',
            help='Whether to add normal and depth outputs '\
                    + 'for ImageNet and PlaceNet, ' \
                    + 'default is 0, which means no')
    parser.add_argument(
	    '--corr_bypassadd', default=0, type=int, action='store',
            help='Whether use the correct bypass add method')
    parser.add_argument(
	    '--ignorebname', default=0, type=int, action='store',
            help='Whether ignore the batch name')
    parser.add_argument(
	    '--ignorebname_new', default=1, type=int, action='store',
            help='Whether ignore the batch name in conv and resblock')
    parser.add_argument(
	    '--add_batchname', default=None, type=str, action='store',
            help='Batch name used in conv and resblock, '\
                    + 'when None, using default')
    parser.add_argument(
	    '--train_anyway', default=0, type=int, action='store',
            help='Whether make the bn in train stage anyway')

    ## Mean teacher related
    parser.add_argument(
	    '--mean_teacher', default=0, type=int, action='store',
            help='Whether use the mean teacher setting')
    parser.add_argument(
	    '--res_coef', default=0.01, type=float, action='store',
            help='Coefficient for l2 loss between con and class logits')
    parser.add_argument(
	    '--cons_ramp_len', default=100000, type=int, action='store',
            help='Ramp up length for consistency coefficient')
    parser.add_argument(
	    '--cons_max_value', default=10, type=float, action='store',
            help='Max value for consistency coefficient')
    parser.add_argument(
	    '--ema_decay', default=0.9997, type=float, action='store',
            help='Teacher decay used in mean teacher setting')
    parser.add_argument(
	    '--ema_zerodb', default=0, type=int, action='store',
            help='Whether zero debias the ema, default is 0 (none)')
    parser.add_argument(
	    '--mt_ckpt_load_dict', default=0, type=int, action='store',
            help='Whether setting load_param_dict '\
                    + 'by searching variables in ckpt')
    parser.add_argument(
	    '--mt_infant_loss', default=0, type=int, action='store',
            help='Whether set one hot label to be 246 dimension')

    ## Instance related
    parser.add_argument(
	    '--instance_task', 
            action='store_true',
            help='Whether doing the instance task, default is 0 (no)')
    parser.add_argument(
	    '--instance_k', default=4096, type=int, action='store',
            help='The number of noisy examples chosen for instance task, '\
                    + 'default is 4096')
    parser.add_argument(
	    '--instance_t', default=0.07, type=float, action='store',
            help='The temparature for instance task')
    parser.add_argument(
	    '--instance_m', default=0.5, type=float, action='store',
            help='The m for instance task')
    parser.add_argument(
	    '--instance_data_len', default=1281167, type=int, action='store',
            help='Length of the dataset, needed to build the memory bank')
    parser.add_argument(
	    '--inst_lbl_pkl', default=None, type=str, action='store',
            help='Label pkl file for instance task')
    parser.add_argument(
	    '--inst_cate_sep', 
            action='store_true',
            help='Doing instance and categorization')
    parser.add_argument(
	    '--inst_clstr_path', default=None, type=str, action='store',
            help='Label npy file for instance clustering task')
    parser.add_argument(
	    '--imgnt_w_idx', action='store_true',
            help='Whether load index from tfrecords for ImageNet')
    parser.add_argument(
	    '--semi_clstr_path', default=None, type=str, action='store',
            help='Label npy file for instance semi clustering task')
    parser.add_argument(
	    '--semi_name_scope', default=None, type=str, action='store',
            help='Name scope prefix for semi learning')

    ## Implemented by Siming
    parser.add_argument(
	    '--sm_fix', default=0, type=int, action='store',
            help='Whether fix the encoder parameters '\
                    + 'when training on the ImageNet, '\
                    + 'default is 0, which means not fix. '\
                    + '1 means fix the whole encoder path, '\
                    + '2 means release the lasts resblock of encoder path, '\
                    + '3 means release the last two, '\
                    + '4 means release the last three')  # sm_add
    parser.add_argument(
	    '--sm_de_fix', default=0, type=int, action='store',
            help="Whether fix the decoder parameters")  # sm_add
    parser.add_argument(
	    '--sm_depth_fix', default=0, type=int, action='store',
            help="Whether fix the depth parameters")  # sm_add
    parser.add_argument(
	    '--sm_resnetv2', default=0, type=int, action='store',
            help="Whether use resnet v2")  # sm_add
    parser.add_argument(
	    '--sm_resnetv2_1', default=0, type=int, action='store',
            help="Whether use resnet v2_1")  # sm_add
    parser.add_argument(
	    '--sm_bn_trainable', default=1, type=int, action='store',
            help="Whether train the batch normalization layer")  # sm_add
    parser.add_argument(
	    '--sm_bn_fix', default=0, type=int, action='store',
            help="Whether fix the encoder bn, similar with sm_fix")  # sm_add
    return parser


def add_loss_setting(parser):
    # Loss related
    parser.add_argument(
	    '--depth_norm', default=8000, type=int, action='store',
            help='Coefficient for depth loss')
    parser.add_argument(
	    '--label_norm', default=1, type=float, action='store',
            help='Coefficient for label loss')
    parser.add_argument(
	    '--depthloss', default=0, type=int, action='store',
            help='Whether to use new depth loss')
    parser.add_argument(
	    '--normalloss', default=0, type=int, action='store',
            help='Whether to use new normal loss')
    parser.add_argument(
	    '--multtime', default=1, type=int, action='store',
            help='1 means original, larger than 1 means multiple time points')
    parser.add_argument(
	    '--sm_half_size', default=0, type=int, action='store',
            help='whether output halfsize of depth & normal maps')
    return parser


def add_train_setting(parser):
    # Training related
    parser.add_argument(
	    '--batchsize', default=None, type=int, action='store',
            help='Batch size')
    parser.add_argument(
	    '--valbatchsize', default=None, type=int, action='store',
            help='Validation Batch size')
    parser.add_argument(
	    '--queuecap', default=None, type=int, action='store',
            help='Queue capacity')
    parser.add_argument(
	    '--init_stddev', default=.01, type=float, action='store',
            help='Init stddev for convs')
    parser.add_argument(
	    '--init_type', default='xavier', type=str, action='store',
            help='Init type')
    parser.add_argument(
	    '--train_num_steps', default=None, type=int, action='store',
            help='Number of training steps')

    # Learning rate, optimizers
    parser.add_argument(
	    '--init_lr', default=.01, type=float, action='store',
            help='Init learning rate')
    parser.add_argument(
	    '--lr_boundaries', default=None, type=str, action='store',
            help='Learning rate drop boundaries')
    parser.add_argument(
	    '--whichopt', default=0, type=int, action='store',
            help='Choice of the optimizer, 0 means momentum, 1 means Adam')
    parser.add_argument(
	    '--trainable_scope', default=None, type=str, action='store',
            help='Trainable scope')
    parser.add_argument(
	    '--adameps', default=0.1, type=float, action='store',
            help='Epsilon for adam, only used when whichopt is 1')
    parser.add_argument(
	    '--adambeta1', default=0.9, type=float, action='store',
            help='Beta1 for adam, only used when whichopt is 1')
    parser.add_argument(
	    '--adambeta2', default=0.999, type=float, action='store',
            help='Beta2 for adam, only used when whichopt is 1')
    parser.add_argument(
	    '--withclip', default=1, type=int, action='store',
            help='Whether do clip')
    parser.add_argument(
	    '--clip_num', default=1, type=float, action='store',
            help='The range of gradient clip')
    parser.add_argument(
	    '--target_lr', default=.025, type=float, action='store',
            help='Only used in mean-teacher tpu model '\
                    + 'as target lr after ramping up, '\
                    + 'target_lr will be mulitplied by BATCH_SIZE/16')
    parser.add_argument(
	    '--mt_ramp_down', default=0, type=int, action='store',
            help='Only used in mean-teacher tpu model to \
                    enable ramp down for learning rate, default is no')
    parser.add_argument(
	    '--mt_ramp_down_epoch', default=75, type=int, action='store',
            help='Only used in mean-teacher tpu model to '\
                    + 'set the ramp down length for learning rate')

    # GPU related
    parser.add_argument(
	    '--gpu', default='0', type=str, action='store',
            help='Availabel GPUs')
    parser.add_argument(
	    '--n_gpus', default=None, type=int, action='store',
            help='Number of GPUs to use, default is None, to use length in gpu')
    parser.add_argument(
	    '--gpu_offset', default=0, type=int, action='store',
            help='Offset of gpu index')
    parser.add_argument(
	    '--minibatch', default=None, type=int, action='store',
            help='Minibatch to use, default to be None, not using')
    return parser


def add_data_setting(parser):
    # Dataset related
    parser.add_argument(
	    '--use_dataset_inter', action='store_true',
            help='Whether use dataset interface')
    parser.add_argument(
	    '--whichimagenet', default='full', type=str, action='store',
            help='Choice of the imagenet')
    parser.add_argument(
	    '--whichcoco', default=0, type=int, action='store',
            help='Which coco dataset to use, \
                    0 means original, 1 means the one without 0 instance')
    parser.add_argument(
	    '--which_place', default=0, type=int, action='store',
            help='Which place dataset to use, 1 means only part')
    parser.add_argument(
	    '--localimagenet', default=None, type=str, action='store',
            help='Home path of imagenet, None means using default')
    parser.add_argument(
	    '--overall_local', default=None, type=str, action='store',
            help='Home path of all datasets, None means using default')

    ## Preprocessing related
    parser.add_argument(
	    '--depthnormal', default=0, type=int, action='store',
            help='Whether to normalize the depth input')
    parser.add_argument(
	    '--depthnormal_div', default=None, type=float, action='store',
            help='Divided value before per_image_standardization')
    parser.add_argument(
	    '--withflip', default=0, type=int, action='store',
            help='Whether flip the input images horizontally')
    parser.add_argument(
	    '--shuffle_seed', default=0, type=int, action='store',
            help='Shuffle seed for the data provider')
    parser.add_argument(
	    '--weight_decay', default=None, type=float, action='store',
            help='Weight decay')
    parser.add_argument(
	    '--global_weight_decay', default=None, type=float, action='store',
            help='Global weight decay (inlcuding the BN) (seems not used now)')
    parser.add_argument(
	    '--enable_bn_weight_decay', action='store_true',
            help='Enable batch normalization weight decay')
    parser.add_argument(
	    '--no_shuffle', default=0, type=int, action='store',
            help='Whether do the shuffling')
    parser.add_argument(
	    '--no_prep', default=0, type=int, action='store',
            help='Avoid the scaling in model function or not')
    parser.add_argument(
	    '--with_color_noise', default=0, type=int, action='store',
            help='Whether do color jittering')
    parser.add_argument(
	    '--color_norm', default=0, type=int, action='store',
            help='Whether doing color normalize')
    parser.add_argument(
	    '--size_vary_prep', default=0, type=int, action='store',
            help='Whether varying size of input images')
    parser.add_argument(
	    '--fix_asp_ratio', default=0, type=int, action='store',
            help='Whether fixing aspect ratio of cropped input images')
    parser.add_argument(
	    '--img_size_vary_prep', default=0, type=int, action='store',
            help='Whether varying size of input images, only for imagenet')
    parser.add_argument(
	    '--sm_full_size', default=0, type=int, action='store',
            help='Input full size image when train on depth and normal dataset')
    parser.add_argument(
	    '--crop_size', default=224, type=int, action='store',
            help='the size of cropping')
    parser.add_argument(
	    '--prob_gray', default=None, type=float, action='store',
            help='If not None, should be between (0,1), '\
                    + 'image will be transformed to grayscale '\
                    + 'according to this number')
    parser.add_argument(
	    '--size_minval', default=0.08, type=float, action='store',
            help='Set the minimal resize area ratio for image preprocessing')

    ## Related to kinetics
    parser.add_argument(
	    '--crop_time', default=5, type=int, action='store',
            help='Crop time for kinetics dataset')
    parser.add_argument(
	    '--crop_rate', default=5, type=int, action='store',
            help='Crop rate for kinetics dataset')
    parser.add_argument(
	    '--replace_folder_train', default=None, type=str, action='store',
            help='Replace_folder for train group')
    parser.add_argument(
	    '--replace_folder_val', default=None, type=str, action='store',
            help='Replace folder for val group')
    return parser


def add_rp_col_setting(parser):
    # RP related
    parser.add_argument(
	    '--num_grids', default=4, type=int, action='store',
            help='How many grids are extracted from images')
    parser.add_argument(
	    '--g_noise', default=0, type=int, action='store',
            help='Whether apply color dropping with gaussian noise')
    parser.add_argument(
	    '--rp_std', default=0, type=int, action='store',
            help='Whether use standard deviation for rp image preprocessing')
    parser.add_argument(
	    '--rp_zip', default=0, type=int, action='store',
            help='Whether use zip version ps dataset')
    parser.add_argument(
	    '--rp_dp_tl', default=0, type=int, action='store',
            help='Do the relative position depth prediction transfer learning')
    parser.add_argument(
	    '--use_lasso', default=0, type=int, action='store',
            help='use lasso in the combine self-supervised learning task')
    parser.add_argument(
	    '--rp_grayscale', default=0, type=int, action='store',
            help='whether input grayscale images')  
    parser.add_argument(
	    '--rp_sub_mean', default=1, type=int, action='store',
            help='only equals 0 when rp_grayscale equals 1')  

    # Colorization related
    parser.add_argument(
	    '--col_down', default=8, type=int, action='store',
            help='The down sample rate of colorization task')
    parser.add_argument(
	    '--gpu_task', default=None, type=str, action='store',
            help='Whether do the colorization on GPU')
    parser.add_argument(
	    '--col_size', default=0, type=int, action='store',
            help='increase the size of validation image')
    parser.add_argument(
	    '--col_knn', default=0, type=int, action='store',
            help='whether use knn to get labels')
    parser.add_argument(
	    '--col_tl', default=0, type=int, action='store',
            help='whether do the colorization transfer learning on tpu')
    parser.add_argument(
	    '--color_dp_tl', default=0, type=int, action='store',
            help='Do the colorization depth prediction transfer learning')
    parser.add_argument(
	    '--combine_col_rp', default=0, type=int, action='store',
            help='whether replicate 3 grayscale channels on gpu')
    parser.add_argument(
	    '--combine_rp', default=0, type=int, action='store',
            help='whether replicate 3 grayscale channels')
    parser.add_argument(
	    '--combine_input_same', default=0, type=int, action='store',
            help='whether input same images')
    parser.add_argument(
	    '--combine_task_imn_tl', default=0, type=int, action='store',
            help='Few shot learning L channel input')
    parser.add_argument(
	    '--input_mode', default='rgb', type=str, action='store',
            help='Input mode, rgb or sobel')
    return parser


def add_tpu_setting(parser):
    # Depth TPU related
    parser.add_argument(
	    '--depth_zip', default=0, type=int, action='store',
            help='whether use depth_zip version on tpu')
    parser.add_argument(
	    '--ab_depth', default=0, type=int, action='store',
            help='whether use absolute depth map')
    parser.add_argument(
	    '--tpu_depth', default=0, type=int, action='store',
            help='Whether do the depth prediction on gpu using tpu model')
    parser.add_argument(
	    '--depth_down', default=1, type=int, action='store',
            help='Whether do the down sample depth map')
    parser.add_argument(
	    '--depth_imn_tl', default=0, type=int, action='store',
            help='Whether do the depth imagenet transfer learning on TPU')

    # TPU related
    parser.add_argument(
	    '--tpu_name', default='siming-tpu', type=str, action='store',
            help='Tpu name')
    parser.add_argument(
	    '--gcp_project', default=None, type=str, action='store',
            help='Project id')
    parser.add_argument(
	    '--tpu_zone', default=None, type=str, action='store',
            help='Tpu zone name')
    parser.add_argument(
	    '--tpu_task', default=None, type=str, action='store',
            help='Which task do you want to run on the tpu')
    parser.add_argument(
	    '--tpu_num_shards', default=8, type=int, action='store',
            help='Number of shards in tpu')
    parser.add_argument(
	    '--tpu_flag', default=0, type=int, action='store',
            help='The difference between gpu and tpu model funtion')
    parser.add_argument(
	    '--tpu_tl_imagenet', default=None, type=str, action='store',
            help='Whether do the transfer learning task on GPU')
    parser.add_argument(
	    '--combine_tpu_flag', default=0, type=int, action='store',
            help='Combinet on tpu or not, as on tpu all the bn name is same')
    parser.add_argument(
	    '--validation_skip', default=0, type=int, action='store',
            help='Whether skip the validation stage')
    parser.add_argument(
	    '--resnet_prep', action='store_true',
            help='Whether using resnet preprocessing, only crop and flip')
    parser.add_argument(
	    '--resnet_prep_size', action='store_true',
            help='Whether using resnet preprocessing for cropping min size')
    return parser


def add_pca_setting(parser):
    # PCA related
    parser.add_argument(
	    '--do_pca', action='store_true',
            help='Whether do pca validation')
    parser.add_argument(
	    '--pca_load_setting_func', default=None, type=str, 
            action='store',
            help='Setting name for pca')
    parser.add_argument(
	    '--pca_n_components', default=1000, type=int, 
            action='store',
            help='Number of components for PCA')
    parser.add_argument(
	    '--pca_save_dir', 
            default='/mnt/fs4/chengxuz/v4it_temp_results/pca_results', 
            type=str, action='store',
            help='Directory to save pca results')
    return parser


def get_parser():
    parser = argparse.ArgumentParser(
            description='The script to train the combine net')

    parser = add_general_setting(parser)
    parser = add_network_setting(parser)
    parser = add_loss_setting(parser)
    parser = add_train_setting(parser)
    parser = add_data_setting(parser)
    parser = add_rp_col_setting(parser)
    parser = add_tpu_setting(parser)
    parser = add_pca_setting(parser)
    return parser


def load_setting(args):
    if args.load_setting_func:
        setting_func = get_setting_func(args.load_setting_func)
        args = setting_func(args)
    return args
