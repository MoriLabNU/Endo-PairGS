ModelParams = dict(
    camera_extent = 10,
    use_pretrain= True, #True
    init_pcd_nums= 2,
    num_gaussians = 25, # 50
)
OptimizationParams = dict(

    coarse_iterations = 1000,
    deformation_lr_init = 0.00016, # 0.00016
    deformation_lr_final = 0.0000016,
    deformation_lr_delay_mult = 0.01,
    grid_lr_init = 0.0016,
    grid_lr_final = 0.000016,
    iterations = 10000,
    percent_dense = 0.01,
    render_process=True,
    densify_until_iter = 15_000,
    pruning_from_iter = 500,
    densify_from_iter = 500,
    # lambda_dssim=0.2,
    densification_interval=100,
    pruning_interval = 100,
    # add_point=True,
    # no_do=False,
    # no_dshs=False,
    opacity_reset_interval=12000,
    c2f = True,
    c2f_every_step = 1000,
    c2f_max_lowpass = 300,
)
ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 32,
     'resolution': [64, 64, 64, 75]
    },
    multires = [1, 2],
    defor_depth = 0, # original 0, ours 8
    net_width = 64,
    plane_tv_weight = 0.0001,
    time_smoothness_weight = 0.01,
    l1_time_planes =  0.0001,
    depth_weight = 0.01, # 200, 0.01
    grad_weight=0.001,
    normal_weight = 0.001,
    un_img_weight = 0.001,
    un_dep_weight = 0.001,
    weight_decay_iteration=0,
    bounds=1.6,
    pool_list=[2],
    multi_scale=False
)

PipelineParams = dict(
    # use_depth=False,
    use_smooth=False,
    use_normal=False,
    use_depth=True, #False,
    # use_smooth=True, #False,
    # use_normal=True, #False,
    use_confidence=True,
)
