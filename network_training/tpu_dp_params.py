from tpu_old_dps.full_imagenet_input import ImageNetInput
from tpu_old_dps.rp_imagenet_input import RP_ImageNetInput
from tpu_old_dps.rp_pbrscenenet_input import PBRSceneNetDepthMltInput
from tpu_old_dps.col_imagenet_input import Col_ImageNetInput
from tpu_old_dps.col_pbrscenenet_input import Col_PBRSceneNetInput
from tpu_old_dps.col_pbr_input import Col_PBRNetInput
from tpu_old_dps.depth_pbrscenenet_input import PBRSceneNetDepthInput
from tpu_old_dps.depth_pbr_input import PBRNetDepthInput
from tpu_old_dps.rp_pbr_input import PBRNetDepthMltInput
from tpu_old_dps.rp_ps_zip_input import PBRSceneNetZipInput
from tpu_old_dps.depth_pbr_zip_input import PBRNetZipDepthInput
from tpu_old_dps.depth_ps_zip_input import PBRSceneNetZipDepthInput
from tpu_old_dps.col_tl_imagenet_input import Col_Tl_Input
from tpu_old_dps.combine_depth_imn_input import DepthImagenetInput
from tpu_old_dps.combine_rp_imagenet_input import Combine_RP_ImageNet_Input 
from tpu_old_dps.combine_rp_col_input import Combine_RP_Color_Input
from tpu_old_dps.combine_rci_input import Combine_RCI_Input
from tpu_old_dps.combine_rp_col_ps_input import Combine_RP_Color_PS_Input
from tpu_old_dps.combine_rp_col_input_new import Combine_RP_Color_Input_New
from tpu_old_dps.combine_rdc_input import Combine_RDC_Input
from tpu_old_dps.combine_rdc_imn_input import Combine_RDC_ImageNet_Input
from tpu_data_provider import TPUCombineWorld
from utilities.data_path_utils import get_TPU_data_path


def get_deprecated_val_tpu_topn_dp_params(args):
    if args.tpu_task=='imagenet_rp':
        val_input_fn = ImageNetInput(False, args.sm_loaddir, std=False).input_fn

    if args.tpu_task=='rp':
        val_input_fn = RP_ImageNetInput(False, args.sm_loaddir).input_fn

    if args.tpu_task=='rp_pbr':
        val_input_fn =  PBRSceneNetZipInput(
                False, args.sm_loaddir, args.sm_loaddir2).input_fn
        if args.rp_zip==0:
            val_input_fn = PBRSceneNetDepthMltInput(
                    False, args.sm_loaddir, args.sm_loaddir2).input_fn

    if args.tpu_task=='rp_only_pbr':
        val_input_fn = PBRNetDepthMltInput(False, args.sm_loaddir).input_fn

    if args.tpu_task=='colorization':
        val_input_fn = Col_ImageNetInput(
                False, args.sm_loaddir, down_sample=args.col_down, 
                col_knn=(args.col_knn==1), col_tl=(args.col_tl==1)).input_fn

    if args.tpu_task=='color_ps':
        val_input_fn = Col_PBRSceneNetInput(
                False, args.sm_loaddir, args.sm_loaddir2, 
                down_sample=args.col_down, col_knn=(args.col_knn==1)).input_fn

    if args.tpu_task=='color_pbr':
        val_input_fn = Col_PBRNetInput(
                False, args.sm_loaddir, 
                down_sample=args.col_down, col_knn=(args.col_knn==1)).input_fn

    if args.tpu_task=='color_tl':
        val_input_fn = Col_Tl_Input(
                False, args.sm_loaddir, 
                down_sample=args.col_down, col_knn=(args.col_knn==1), 
                combine_rp=(args.combine_rp==1)).input_fn

    if args.tpu_task=='depth':
        val_input_fn = PBRSceneNetZipDepthInput(
                False, args.sm_loaddir, args.sm_loaddir2, 
                ab_depth=(args.ab_depth==1), down_sample=args.depth_down, 
                color_dp_tl=(args.color_dp_tl==1), rp_dp_tl=(args.rp_dp_tl==1), 
                rpcol_dp_tl=(args.combine_col_rp==1)).input_fn

        if args.depth_zip==0:
            val_input_fn = PBRSceneNetDepthInput(
                    False, args.sm_loaddir, args.sm_loaddir2).input_fn

    if args.tpu_task=='depth_pbr':
        val_input_fn = PBRNetZipDepthInput(False, args.sm_loaddir).input_fn

        if args.depth_zip==0:
            val_input_fn = PBRNetDepthInput(False, args.sm_loaddir).input_fn

    if args.tpu_task=='combine_depth_imn':
        val_input_fn = DepthImagenetInput(
                False, args.sm_loaddir, args.sm_loaddir2).input_fn

    if args.tpu_task=='combine_rp_imn':
        val_input_fn = Combine_RP_ImageNet_Input(False, args.sm_loaddir).input_fn

    if args.tpu_task=='combine_rp_col':
        val_input_fn = Combine_RP_Color_Input(
                False, args.sm_loaddir, num_grids=1).input_fn
    
    if args.tpu_task=='combine_rci':
        val_input_fn = Combine_RCI_Input(False, args.sm_loaddir, num_grids=1).input_fn
    
    if args.tpu_task=='combine_rp_col_ps':
        val_input_fn = Combine_RP_Color_PS_Input(
                False, args.sm_loaddir, args.sm_loaddir2, num_grids=1).input_fn

    if args.tpu_task=='combine_rdc':
        val_input_fn = Combine_RDC_Input(
                False, args.sm_loaddir, args.sm_loaddir2).input_fn
    if args.tpu_task=='combine_rdc_imn':
        val_input_fn = Combine_RDC_ImageNet_Input(
                False, args.sm_loaddir, args.sm_loaddir2, args.sm_loaddir3).input_fn
    return val_input_fn


def get_deprecated_tpu_train_dp_params(args):
    if args.tpu_task=='imagenet_rp':
        data_provider_func = ImageNetInput(
                True, args.sm_loaddir, 
                std=False).input_fn
    if args.tpu_task=='rp':
        data_provider_func = RP_ImageNetInput(
                True, args.sm_loaddir, 
                g_noise=args.g_noise, std=(args.rp_std==1), 
                sub_mean=(args.rp_sub_mean==1), 
                grayscale=(args.rp_grayscale==1)).input_fn
    if args.tpu_task=='rp_pbr':
        data_provider_func = PBRSceneNetZipInput(
                True, args.sm_loaddir, 
                args.sm_loaddir2, g_noise=args.g_noise, 
                std=(args.rp_std==1)).input_fn
        if args.rp_zip==0:
            data_provider_func = PBRSceneNetDepthMltInput(
                    True, args.sm_loaddir, 
                    args.sm_loaddir2, g_noise=args.g_noise, 
                    std=(args.rp_std==1)).input_fn
    if args.tpu_task=='rp_only_pbr':
        data_provider_func = PBRNetDepthMltInput(
                True, args.sm_loaddir, 
                g_noise=args.g_noise, std=(args.rp_std==1)).input_fn
    if args.tpu_task=='colorization':
        data_provider_func = Col_ImageNetInput(
                True, args.sm_loaddir, 
                down_sample=args.col_down, col_knn=args.col_knn==1, 
                col_tl=(args.col_tl==1), 
                combine_rp=(args.combine_rp==1)).input_fn
    if args.tpu_task=='color_ps':
         data_provider_func = Col_PBRSceneNetInput(
                 True, args.sm_loaddir, 
                 args.sm_loaddir2, down_sample=args.col_down, 
                 col_knn=args.col_knn==1).input_fn  
    if args.tpu_task=='color_pbr':
         data_provider_func = Col_PBRNetInput(
                 True, args.sm_loaddir, 
                 down_sample=args.col_down, 
                 col_knn=args.col_knn==1).input_fn 
    if args.tpu_task=='color_tl':
         data_provider_func = Col_Tl_Input(
                 True, args.sm_loaddir, 
                 down_sample=args.col_down, 
                 col_knn=args.col_knn==1, 
                 combine_rp=(args.combine_rp==1)).input_fn 
    if args.tpu_task=='depth':
        data_provider_func = PBRSceneNetZipDepthInput(
                True, args.sm_loaddir, args.sm_loaddir2, 
                ab_depth=(args.ab_depth==1), down_sample=args.depth_down, 
                color_dp_tl=(args.color_dp_tl==1), 
                rp_dp_tl=(args.rp_dp_tl==1), 
                rpcol_dp_tl=(args.combine_col_rp==1)).input_fn
        if args.depth_zip== 0:
            data_provider_func = PBRSceneNetDepthInput(
                    True, args.sm_loaddir, args.sm_loaddir2).input_fn
    if args.tpu_task=='combine_rp_imn':
        data_provider_func = Combine_RP_ImageNet_Input(
                True, args.sm_loaddir).input_fn
    if args.tpu_task=='combine_rp_col':
        data_provider_func = Combine_RP_Color_Input(
                True, args.sm_loaddir, num_grids=1).input_fn
    if args.tpu_task=='combine_rci':
        data_provider_func = Combine_RCI_Input(
                True, args.sm_loaddir, 
                num_grids=1).input_fn
    if args.tpu_task=='combine_rp_col_ps':
        data_provider_func = Combine_RP_Color_PS_Input(
                True, args.sm_loaddir, 
                args.sm_loaddir2, num_grids=1).input_fn
    if args.tpu_task=='combine_rdc':
        data_provider_func = Combine_RDC_Input(
                True, args.sm_loaddir, 
                args.sm_loaddir2).input_fn
    if args.tpu_task=='combine_rdc_imn':
        data_provider_func = Combine_RDC_ImageNet_Input(
                True, args.sm_loaddir, 
                args.sm_loaddir2, args.sm_loaddir3).input_fn

    return data_provider_func 
