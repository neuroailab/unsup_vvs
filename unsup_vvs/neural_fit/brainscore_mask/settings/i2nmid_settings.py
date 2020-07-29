import json
import settings.rp_settings as rp_settings
import settings.color_settings as color_settings
import settings.depth_settings as depth_settings


def mid_basics(args):
    args.load_step = 50 * 10010
    cfg_kwargs = {
            'module_name': ['encode', 'category_trans'],
            }
    old_cfg_kwargs = json.loads(getattr(args, 'cfg_kwargs', '{}'))
    cfg_kwargs.update(old_cfg_kwargs)
    args.cfg_kwargs = json.dumps(cfg_kwargs)
    args.layers = 'category_trans_1'
    return args


def la_seed0(args):
    args.setting_name = 'imgnt_mid_trans.la_s0'
    args = mid_basics(args)
    return args


def la_seed1(args):
    args.setting_name = 'imgnt_mid_trans.la_s1'
    args = mid_basics(args)
    return args


def la_seed2(args):
    args.setting_name = 'imgnt_mid_trans.la_s2'
    args = mid_basics(args)
    return args


def ir_seed0(args):
    args.setting_name = 'imgnt_mid_trans.ir_s0'
    args = mid_basics(args)
    return args


def ir_seed1(args):
    args.setting_name = 'imgnt_mid_trans.ir_s1'
    args = mid_basics(args)
    return args


def ir_seed2(args):
    args.setting_name = 'imgnt_mid_trans.ir_s2'
    args = mid_basics(args)
    return args


def ae_seed0(args):
    args.setting_name = 'imgnt_mid_trans.ae_s0'
    args = mid_basics(args)
    return args


def ae_seed1(args):
    args.setting_name = 'imgnt_mid_trans.ae_s1'
    args = mid_basics(args)
    return args


def ae_seed2(args):
    args.setting_name = 'imgnt_mid_trans.ae_s2'
    args = mid_basics(args)
    return args


def untrn_seed0(args):
    args.setting_name = 'imgnt_mid_trans.untrn_s0'
    args = mid_basics(args)
    return args


def untrn_seed1(args):
    args.setting_name = 'imgnt_mid_trans.untrn_s1'
    args = mid_basics(args)
    return args


def untrn_seed2(args):
    args.setting_name = 'imgnt_mid_trans.untrn_s2'
    args = mid_basics(args)
    return args


def depth_seed0(args):
    args.setting_name = 'imgnt_mid_trans.depth_s0'
    args = depth_settings.add_cfg_kwargs(args)
    args = mid_basics(args)
    return args


def depth_seed1(args):
    args.setting_name = 'imgnt_mid_trans.depth_s1'
    args = depth_settings.add_cfg_kwargs(args)
    args = mid_basics(args)
    return args


def depth_seed2(args):
    args.setting_name = 'imgnt_mid_trans.depth_s2'
    args = depth_settings.add_cfg_kwargs(args)
    args = mid_basics(args)
    return args


def cpc_seed0(args):
    args.setting_name = 'imgnt_mid_trans.cpc_s0'
    args = mid_basics(args)
    return args


def cpc_seed1(args):
    args.setting_name = 'imgnt_mid_trans.cpc_s1'
    args = mid_basics(args)
    return args


def cpc_seed2(args):
    args.setting_name = 'imgnt_mid_trans.cpc_s2'
    args = mid_basics(args)
    return args


def color_seed0(args):
    args.setting_name = 'imgnt_mid_trans.color_s0'
    args = color_settings.add_cfg_kwargs(args)
    args = mid_basics(args)
    return args


def color_seed1(args):
    args.setting_name = 'imgnt_mid_trans.color_s1'
    args = color_settings.add_cfg_kwargs(args)
    args = mid_basics(args)
    return args


def color_seed2(args):
    args.setting_name = 'imgnt_mid_trans.color_s2'
    args = color_settings.add_cfg_kwargs(args)
    args = mid_basics(args)
    return args


def rp_seed0(args):
    args.setting_name = 'imgnt_mid_trans.rp_s0'
    args = rp_settings.add_cfg_kwargs(args)
    args = mid_basics(args)
    return args


def rp_seed1(args):
    args.setting_name = 'imgnt_mid_trans.rp_s1'
    args = rp_settings.add_cfg_kwargs(args)
    args = mid_basics(args)
    return args


def rp_seed2(args):
    args.setting_name = 'imgnt_mid_trans.rp_s2'
    args = rp_settings.add_cfg_kwargs(args)
    args = mid_basics(args)
    return args


def simclr_seed0(args):
    args.setting_name = 'imgnt_mid_trans.simclr_prep'
    args = mid_basics(args)
    args.model_type = 'simclr_model_mid'
    return args


def simclr_seed1(args):
    args.setting_name = 'imgnt_mid_trans.simclr_prep_seed1'
    args = mid_basics(args)
    args.model_type = 'simclr_model_mid'
    return args


def simclr_seed2(args):
    args.setting_name = 'imgnt_mid_trans.simclr_prep_seed2'
    args = mid_basics(args)
    args.model_type = 'simclr_model_mid'
    return args


PT_MID_MODEL_CACHE_FOLDER = '/mnt/fs4/chengxuz/v4it_temp_results/.result_caching/pt_mid_model_imagenet_transfer/'
def dc_seed0(args):
    args.pt_model = 'deepcluster'
    args.load_from_ckpt = '/mnt/fs4/chengxuz/deepcluster_models/res18_np/checkpoint.pth.tar'
    args.identifier = 'pt-dc-mid-seed0'
    args.pt_model_mid = PT_MID_MODEL_CACHE_FOLDER + 'pt-dc-seed0/checkpoints/checkpoint_69.pth.tar'
    return args


def dc_seed1(args):
    args.pt_model = 'deepcluster'
    args.load_from_ckpt = '/mnt/fs4/chengxuz/deepcluster_models/res18_np_seed0/checkpoint.pth.tar'
    args.identifier = 'pt-dc-mid-seed1'
    args.pt_model_mid = PT_MID_MODEL_CACHE_FOLDER + 'pt-dc-seed1/checkpoints/checkpoint_69.pth.tar'
    return args


def dc_seed2(args):
    args.pt_model = 'deepcluster'
    args.load_from_ckpt = '/mnt/fs4/chengxuz/deepcluster_models/res18_np_seed1/checkpoint.pth.tar'
    args.identifier = 'pt-dc-mid-seed2'
    args.pt_model_mid = PT_MID_MODEL_CACHE_FOLDER + 'pt-dc-seed2/checkpoints/checkpoint_69.pth.tar'
    return args


def cmc_seed0(args):
    args.pt_model = 'la_cmc'
    args.load_from_ckpt = '/mnt/fs4/chengxuz/la_cmc_models/experiments/imagenet_la_cmc/res18_lab_cmc_v1_s0/checkpoints/final.pth.tar'
    args.identifier = 'pt-cmc-mid-seed0'
    args.pt_model_mid = PT_MID_MODEL_CACHE_FOLDER + 'pt-cmc-seed0/checkpoints/checkpoint_69.pth.tar'
    return args


def cmc_seed1(args):
    args.pt_model = 'la_cmc'
    args.load_from_ckpt = '/mnt/fs4/chengxuz/la_cmc_models/experiments/imagenet_la_cmc/res18_lab_cmc_v1_s1/checkpoints/final.pth.tar'
    args.identifier = 'pt-cmc-mid-seed1'
    args.pt_model_mid = PT_MID_MODEL_CACHE_FOLDER + 'pt-cmc-seed1/checkpoints/checkpoint_69.pth.tar'
    return args


def cmc_seed2(args):
    args.pt_model = 'la_cmc'
    args.load_from_ckpt = '/mnt/fs4/chengxuz/la_cmc_models/experiments/imagenet_la_cmc/res18_lab_cmc_v1_s2/checkpoints/final.pth.tar'
    args.identifier = 'pt-cmc-mid-seed2'
    args.pt_model_mid = PT_MID_MODEL_CACHE_FOLDER + 'pt-cmc-seed2/checkpoints/checkpoint_69.pth.tar'
    return args
