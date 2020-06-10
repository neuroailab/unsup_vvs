def llp_p03(args):
    args.layers = ','.join(['encode_%i' % i for i in range(2, 11)])
    args.model_type = 'inst_model'
    args.load_step = 4153735
    args.load_port = 27006
    args.load_dbname = 'aggre_semi'
    args.load_colname = 'dyn_clstr'
    args.load_expId = 'p03_tp10_wc_cf_lclw'
    return args


def llp_p03_400(args):
    args = llp_p03(args)
    args.load_step = 400 * 10009
    args.layers = 'encode_10'
    return args


def llp_p03_390(args):
    args = llp_p03(args)
    args.load_step = 390 * 10009
    args.layers = 'encode_10'
    return args


def llp_p03_350(args):
    args = llp_p03(args)
    args.load_step = 350 * 10009
    args.layers = 'encode_10'
    return args


def llp_p01(args):
    args.layers = ','.join(['encode_%i' % i for i in range(2, 11)])
    args.model_type = 'inst_model'
    args.load_step = 228 * 10009
    args.load_port = 27006
    args.load_dbname = 'aggre_semi'
    args.load_colname = 'dyn_clstr'
    args.load_expId = 'p01_tp10_wc_cf_lclw'
    return args


def llp_p01_210(args):
    args = llp_p01(args)
    args.load_step = 210 * 10009
    args.layers = 'encode_10'
    return args


def llp_p01_seed1(args):
    args = llp_p02(args)
    args.load_step = 180 * 10009
    args.load_expId = 'p01_llp_s1'
    args.layers = 'encode_10'
    return args


def llp_p01_cont_ep300_seed1(args):
    args = llp_p02(args)
    args.load_step = 300 * 10009
    args.load_expId = 'p01_llp_cont_s1'
    args.layers = 'encode_10'
    return args


def llp_p01_seed2(args):
    args = llp_p02(args)
    args.load_step = 180 * 10009
    args.load_expId = 'p01_llp_s2'
    args.layers = 'encode_10'
    return args


def llp_p01_cont_ep330_seed2(args):
    args = llp_p02(args)
    args.load_step = 330 * 10009
    args.load_expId = 'p01_llp_cont_s2'
    args.layers = 'encode_10'
    return args


def llp_p05(args):
    args = llp_p01(args)
    args.load_step = 330 * 10009
    args.load_expId = 'p05_tp10_wc_cf_lclw'
    return args


def llp_p10(args):
    args = llp_p01(args)
    args.load_step = 310 * 10009
    args.load_expId = 'p10_tp10_wc_cf_lclw'
    return args


def llp_p02(args):
    args.layers = ','.join(['encode_%i' % i for i in range(2, 11)])
    args.model_type = 'inst_model'
    args.load_step = 330 * 10009
    args.load_port = 26001
    args.load_dbname = 'llp_for_vm'
    args.load_colname = 'dyn_clstr'
    args.load_expId = 'p02_llp'
    return args


def llp_p02_380(args):
    args = llp_p02(args)
    args.load_step = 380 * 10009
    args.layers = 'encode_10'
    return args


def llp_p02_370(args):
    args = llp_p02(args)
    args.load_step = 370 * 10009
    args.layers = 'encode_10'
    return args


def llp_p02_360(args):
    args = llp_p02(args)
    args.load_step = 360 * 10009
    args.layers = 'encode_10'
    return args


def llp_p02_seed1(args):
    args = llp_p02(args)
    args.load_step = 170 * 10009
    args.load_expId = 'p02_llp_s1'
    args.layers = 'encode_10'
    return args


def llp_p02_cont_ep310_seed1(args):
    args = llp_p02(args)
    args.load_step = 310 * 10009
    args.load_expId = 'p02_llp_cont_s1'
    args.layers = 'encode_10'
    return args


def llp_p02_cont_ep350_seed1(args):
    args = llp_p02(args)
    args.load_step = 350 * 10009
    args.load_expId = 'p02_llp_cont_s1'
    args.layers = 'encode_10'
    return args


def llp_p02_seed2(args):
    args = llp_p02(args)
    args.load_step = 170 * 10009
    args.load_expId = 'p02_llp_s2'
    args.layers = 'encode_10'
    return args


def llp_p02_cont_ep310_seed2(args):
    args = llp_p02(args)
    args.load_step = 310 * 10009
    args.load_expId = 'p02_llp_cont_s2'
    args.layers = 'encode_10'
    return args


def llp_p02_cont_ep350_seed2(args):
    args = llp_p02(args)
    args.load_step = 350 * 10009
    args.load_expId = 'p02_llp_cont_s2'
    args.layers = 'encode_10'
    return args


def llp_p03_seed1(args):
    args = llp_p02(args)
    args.load_step = 170 * 10009
    args.load_expId = 'p03_llp_s1'
    args.layers = 'encode_10'
    return args


def llp_p03_cont_ep310_seed1(args):
    args = llp_p02(args)
    args.load_step = 310 * 10009
    args.load_expId = 'p03_llp_cont_s1'
    args.layers = 'encode_10'
    return args


def llp_p03_cont_ep350_seed1(args):
    args = llp_p02(args)
    args.load_step = 350 * 10009
    args.load_expId = 'p03_llp_cont_s1'
    args.layers = 'encode_10'
    return args


def llp_p03_seed2(args):
    args = llp_p02(args)
    args.load_step = 170 * 10009
    args.load_expId = 'p03_llp_s2'
    args.layers = 'encode_10'
    return args


def llp_p03_cont_ep310_seed2(args):
    args = llp_p02(args)
    args.load_step = 310 * 10009
    args.load_expId = 'p03_llp_cont_s2'
    args.layers = 'encode_10'
    return args


def llp_p03_cont_ep350_seed2(args):
    args = llp_p02(args)
    args.load_step = 350 * 10009
    args.load_expId = 'p03_llp_cont_s2'
    args.layers = 'encode_10'
    return args


def llp_p04(args):
    args = llp_p02(args)
    args.load_step = 327 * 10009
    args.load_expId = 'p04_llp'
    return args


def llp_p04_seed1(args):
    args = llp_p02(args)
    args.load_step = 170 * 10009
    args.load_expId = 'p04_llp_s1'
    args.layers = 'encode_10'
    return args


def llp_p04_seed2(args):
    args = llp_p02(args)
    args.load_step = 170 * 10009
    args.load_expId = 'p04_llp_s2'
    args.layers = 'encode_10'
    return args


def llp_p05_seed1(args):
    args = llp_p02(args)
    args.load_step = 170 * 10009
    args.load_expId = 'p05_llp_s1'
    args.layers = 'encode_10'
    return args


def llp_p05_seed2(args):
    args = llp_p02(args)
    args.load_step = 170 * 10009
    args.load_expId = 'p05_llp_s2'
    args.layers = 'encode_10'
    return args


def llp_p06(args):
    args = llp_p02(args)
    args.load_step = 342 * 10009
    args.load_expId = 'p06_llp'
    return args


def llp_p06_seed1(args):
    args = llp_p02(args)
    args.load_step = 170 * 10009
    args.load_expId = 'p06_llp_s1'
    args.layers = 'encode_10'
    return args


def llp_p06_seed2(args):
    args = llp_p02(args)
    args.load_step = 170 * 10009
    args.load_expId = 'p06_llp_s2'
    args.layers = 'encode_10'
    return args


def llp_p10_seed1(args):
    args = llp_p02(args)
    args.load_step = 170 * 10009
    args.load_expId = 'p10_llp_s1'
    args.layers = 'encode_10'
    return args


def llp_p10_seed2(args):
    args = llp_p02(args)
    args.load_step = 170 * 10009
    args.load_expId = 'p10_llp_s2'
    args.layers = 'encode_10'
    return args


def llp_p20(args):
    args = llp_p02(args)
    args.load_step = 315 * 10009
    args.load_expId = 'p20_llp'
    return args


def llp_p20_seed1(args):
    args = llp_p02(args)
    args.load_step = 170 * 10009
    args.load_expId = 'p20_llp_s1'
    args.layers = 'encode_10'
    return args


def llp_p20_seed2(args):
    args = llp_p02(args)
    args.load_step = 170 * 10009
    args.load_expId = 'p20_llp_s2'
    args.layers = 'encode_10'
    return args


def llp_p50(args):
    args = llp_p02(args)
    args.load_step = 287 * 10009
    args.load_expId = 'p50_llp'
    return args


def llp_p50_seed1(args):
    args = llp_p02(args)
    args.load_step = 160 * 10009
    args.load_expId = 'p50_llp_s1'
    args.layers = 'encode_10'
    return args


def llp_p50_seed2(args):
    args = llp_p02(args)
    args.load_step = 160 * 10009
    args.load_expId = 'p50_llp_s2'
    args.layers = 'encode_10'
    return args


def llp_p01_frm_LA(args):
    args = llp_p02(args)
    args.load_step = 332 * 10009
    args.load_expId = 'p01_llp_frmLA'
    return args


def llp_p03_frm_LA(args):
    args = llp_p02(args)
    args.load_step = 332 * 10009
    args.load_expId = 'p03_llp_frmLA'
    return args
