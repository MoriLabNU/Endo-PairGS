import json
import os
import sys

import einops
import lightning as L
import lpips
import omegaconf
import torch
import wandb
import random

# Add MAST3R and PixelSplat to the sys.path to prevent issues during importing
sys.path.append('src/pixelsplat_src')
sys.path.append('src/mast3r_src')
sys.path.append('src/mast3r_src/dust3r')
from src.mast3r_src.dust3r.dust3r.losses import L21
from src.mast3r_src.mast3r.losses import ConfLoss, Regr3D
# import data.scannetpp.scannetpp as scannetpp
import data.endonerf as endonerf
import src.mast3r_src.mast3r.model as mast3r_model
import src.pixelsplat_src.benchmarker as benchmarker
import src.pixelsplat_src.decoder_splatting_cuda as pixelsplat_decoder
import utils.compute_ssim as compute_ssim
import utils.export as export
import utils.geometry as geometry
import utils.loss_mask as loss_mask
import utils.sh_utils as sh_utils
import workspace
from torchvision.models.optical_flow import raft_large,Raft_Large_Weights


class MAST3RGaussians(L.LightningModule):

    def __init__(self, config):

        super().__init__()

        # Save the config
        self.config = config

        # The encoder which we use to predict the 3D points and Gaussians,
        # trained as a modified MAST3R model. The model's configuration is
        # primarily defined by the pretrained checkpoint that we load, see
        # MASt3R's README.md
        self.encoder = mast3r_model.AsymmetricMASt3R(
            pos_embed='RoPE100',
            patch_embed_cls='ManyAR_PatchEmbed',
            img_size=(640, 512),
            head_type='gaussian_head',
            output_mode='pts3d+gaussian+desc24',
            depth_mode=('exp', -mast3r_model.inf, mast3r_model.inf),
            conf_mode=('exp', 1, mast3r_model.inf),
            enc_embed_dim=1024,
            enc_depth=24,
            enc_num_heads=16,
            dec_embed_dim=768,
            dec_depth=12,
            dec_num_heads=12,
            two_confs=True,
            use_offsets=config.use_offsets,
            sh_degree=config.sh_degree if hasattr(config, 'sh_degree') else 1
        )
        self.encoder.requires_grad_(False)
        # self.encoder.downstream_head1.requires_grad_(True)
        # self.encoder.downstream_head2.requires_grad_(True)
        self.encoder.downstream_head1.gaussian_dpt.dpt.requires_grad_(True)
        self.encoder.downstream_head2.gaussian_dpt.dpt.requires_grad_(True)
        # self.encoder.downstream_head1.head_local_features.requires_grad_(True)
        # self.encoder.downstream_head2.head_local_features.requires_grad_(True)

        # The decoder which we use to render the predicted Gaussians into
        # images, lightly modified from PixelSplat
        self.decoder = pixelsplat_decoder.DecoderSplattingCUDA(
            background_color=[0.0, 0.0, 0.0]
        )

        self.benchmarker = benchmarker.Benchmarker()
        
        if self.config.optical_flow:
            self.raft = raft_large(weights=Raft_Large_Weights.DEFAULT)
            self.raft.eval()

        # Loss criteria
        if config.loss.average_over_mask:
            self.lpips_criterion = lpips.LPIPS('vgg', spatial=True)
        else:
            self.lpips_criterion = lpips.LPIPS('vgg')

        if config.loss.mast3r_loss_weight is not None:
            self.mast3r_criterion = ConfLoss(Regr3D(L21, norm_mode='?avg_dis'), alpha=0.2)
            self.encoder.downstream_head1.requires_grad_(True)
            self.encoder.downstream_head2.requires_grad_(True)

        self.save_hyperparameters()

    def forward(self, view1, view2):

        # Freeze the encoder and decoder
        with torch.no_grad():
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = self.encoder._encode_symmetrized(view1, view2)
            dec1, dec2 = self.encoder._decoder(feat1, pos1, feat2, pos2)

        # Train the downstream heads
        pred1 = self.encoder._downstream_head(1, [tok.float() for tok in dec1], shape1, view1["maskmap"])
        pred2 = self.encoder._downstream_head(2, [tok.float() for tok in dec2], shape2, view2["maskmap"])

        pred1['covariances'] = geometry.build_covariance(pred1['scales'], pred1['rotations'])
        pred2['covariances'] = geometry.build_covariance(pred2['scales'], pred2['rotations'])

        learn_residual = True
        if learn_residual:
            new_sh1 = torch.zeros_like(pred1['sh'])
            new_sh2 = torch.zeros_like(pred2['sh'])
            new_sh1[..., 0] = sh_utils.RGB2SH(einops.rearrange(view1['original_img'], 'b c h w -> b h w c'))
            new_sh2[..., 0] = sh_utils.RGB2SH(einops.rearrange(view2['original_img'], 'b c h w -> b h w c'))
            pred1['sh'] = pred1['sh'] + new_sh1
            pred2['sh'] = pred2['sh'] + new_sh2

        # Update the keys to make clear that pts3d and means are in view1's frame
        pred2['pts3d_in_other_view'] = pred2.pop('pts3d')
        pred2['means_in_other_view'] = pred2.pop('means')

        return pred1, pred2

    def training_step(self, batch, batch_idx):

        _, _, h, w = batch["context"][0]["img"].shape
        view1, view2 = batch['context']

        # calculate opticalflow-based mask
        mask_1, mask_2, mask = self.calculate_overlap(view1, view2, batch["target"][0], is_optical=self.config.optical_flow)
        view1["maskmap"] = mask_1
        view2["maskmap"] = mask_2
        # batch['target'][0]["maskmap"] = mask
        
        # Predict using the encoder/decoder and calculate the loss
        pred1, pred2 = self.forward(view1, view2)
        color, _ = self.decoder(batch, pred1, pred2, (h, w))

        # Calculate losses
        # mask = loss_mask.calculate_loss_mask(batch)
        # mask = batch["target"][0]["maskmap"]
        # pre_mask = torch.ones_like(color, device=color.device).cpu().numpy()
        # pre_mask[color.detach().cpu().numpy()<0.03] = 0
        # pre_mask[color.detach().cpu().numpy()<0.003] = 0
        # print(pre_mask.shape)
        # pre_mask = torch.from_numpy(pre_mask[:,:,0,:,:]).cuda()
        # mask = torch.stack([target_view['maskmap'] for target_view in batch['target']], dim=1) 
        
        # mask = torch.logical_and(mask, pre_mask).int()
        # print(mask.shape)
        # print(color.shape)
        loss, mse, lpips = self.calculate_loss(
            batch, view1, view2, pred1, pred2, color, mask,
            apply_mask=self.config.loss.apply_mask,
            average_over_mask=self.config.loss.average_over_mask,
            calculate_ssim=False
        )

        # Log losses
        self.log_metrics('train', loss, mse, lpips)
        return loss

    def validation_step(self, batch, batch_idx):

        _, _, h, w = batch["context"][0]["img"].shape
        view1, view2 = batch['context']

        # calculate opticalflow-based mask
        mask_1, mask_2, mask = self.calculate_overlap(view1, view2, batch["target"][0], is_optical=self.config.optical_flow)
        view1["maskmap"] = mask_1
        view2["maskmap"] = mask_2
        # batch['target'][0]["maskmap"] = mask
        
        # Predict using the encoder/decoder and calculate the loss
        pred1, pred2 = self.forward(view1, view2)
        color, _ = self.decoder(batch, pred1, pred2, (h, w))

        # Calculate losses
        # mask = loss_mask.calculate_loss_mask(batch)
        # mask = batch["target"][0]["maskmap"]
        # pre_mask = torch.ones_like(color, device=color.device).cpu().numpy()
        # pre_mask[color.cpu().numpy()<0.003] = 0
        # print(pre_mask.shape)
        # pre_mask = torch.from_numpy(pre_mask[:,:,0,:,:]).cuda()
        # print(pre_mask.shape)
        # mask = torch.stack([target_view['maskmap'] for target_view in batch['target']], dim=1) 
        # mask = self.calculate_overlap(view1, view2, batch["target"][0])
        # print(mask.shape)
        # mask = torch.logical_and(mask, pre_mask).int()
        # print(mask.shape)
        # print(color.shape)
        # mask = torch.stack([target_view['maskmap'] for target_view in batch['target']], dim=1)
        loss, mse, lpips = self.calculate_loss(
            batch, view1, view2, pred1, pred2, color, mask,
            apply_mask=self.config.loss.apply_mask,
            average_over_mask=self.config.loss.average_over_mask,
            calculate_ssim=False
        )

        # Log losses
        self.log_metrics('val', loss, mse, lpips)
        return loss

    def test_step(self, batch, batch_idx):

        _, _, h, w = batch["context"][0]["img"].shape
        view1, view2 = batch['context']
        num_targets = len(batch['target'])

        # calculate opticalflow-based mask
        mask_1, mask_2, mask = self.calculate_overlap(view1, view2, batch["target"][0], is_optical=self.config.optical_flow)
        view1["maskmap"] = mask_1
        view2["maskmap"] = mask_2
        # batch['target'][0]["maskmap"] = mask
        
        # Predict using the encoder/decoder and calculate the loss
        with self.benchmarker.time("encoder"):
            pred1, pred2 = self.forward(view1, view2)
        with self.benchmarker.time("decoder", num_calls=num_targets):
            color, _ = self.decoder(batch, pred1, pred2, (h, w))

        # Calculate losses
        # mask = loss_mask.calculate_loss_mask(batch)
        # mask = batch["target"][0]["maskmap"]
        # pre_mask = torch.ones_like(color, device=color.device).cpu().numpy()
        # pre_mask[color.detach().cpu().numpy()<0.003] = 0
        # pre_mask = torch.from_numpy(pre_mask[:,:,0,:,:]).cuda()
        # mask = torch.stack([target_view['maskmap'] for target_view in batch['target']], dim=1) 
        # mask = self.calculate_overlap(view1, view2, batch["target"][0])
        # mask = torch.logical_and(mask, pre_mask).int()
        # mask = torch.stack([target_view['maskmap'] for target_view in batch['target']], dim=1)
        loss, mse, lpips, ssim = self.calculate_loss(
            batch, view1, view2, pred1, pred2, color, mask,
            apply_mask=self.config.loss.apply_mask,
            average_over_mask=self.config.loss.average_over_mask,
            calculate_ssim=True
        )

        # Log losses
        self.log_metrics('test', loss, mse, lpips, ssim=ssim)
        return loss

    def on_test_end(self):
        benchmark_file_path = os.path.join(self.config.save_dir, "benchmark.json")
        self.benchmarker.dump(os.path.join(benchmark_file_path))

    def calculate_loss(self, batch, view1, view2, pred1, pred2, color, mask, apply_mask=True, average_over_mask=True, calculate_ssim=False):

        target_color = torch.stack([target_view['original_img'] for target_view in batch['target']], dim=1)
        predicted_color = color

        if apply_mask:
            # assert mask.sum() > 0, "There are no valid pixels in the mask!"
            target_color = target_color * mask[..., None, :, :]
            predicted_color = predicted_color * mask[..., None, :, :]
        # print(target_color.shape)
        flattened_color = einops.rearrange(predicted_color, 'b v c h w -> (b v) c h w')
        flattened_target_color = einops.rearrange(target_color, 'b v c h w -> (b v) c h w')
        flattened_mask = einops.rearrange(mask, 'b v h w -> (b v) h w')
        # flattened_mask = mask

        # MSE loss
        rgb_l2_loss = (predicted_color - target_color) ** 2
        if average_over_mask:
            mse_loss = (rgb_l2_loss * mask[:, None, ...]).sum() / (mask.sum() + 1e-7)
        else:
            mse_loss = rgb_l2_loss.mean()

        # LPIPS loss
        lpips_loss = self.lpips_criterion(flattened_target_color, flattened_color, normalize=True)
        # print(lpips_loss.shape)
        if average_over_mask:
            lpips_loss = (lpips_loss * flattened_mask[:, None, ...]).sum() / (flattened_mask.sum() + 1e-7)
        else:
            lpips_loss = lpips_loss.mean()

        # Calculate the total loss
        loss = 0
        loss += self.config.loss.mse_loss_weight * mse_loss
        loss += self.config.loss.lpips_loss_weight * lpips_loss

        # MAST3R Loss
        if self.config.loss.mast3r_loss_weight is not None:
            mast3r_loss = self.mast3r_criterion(view1, view2, pred1, pred2)[0]
            loss += self.config.loss.mast3r_loss_weight * mast3r_loss

        # Masked SSIM
        if calculate_ssim:
            if average_over_mask:
                ssim_val = compute_ssim.compute_ssim(flattened_target_color, flattened_color, full=True)
                ssim_val = (ssim_val * flattened_mask[:, None, ...]).sum() / (flattened_mask.sum() + 1e-7)
            else:
                ssim_val = compute_ssim.compute_ssim(flattened_target_color, flattened_color, full=False)
                ssim_val = ssim_val.mean()
            return loss, mse_loss, lpips_loss, ssim_val

        return loss, mse_loss, lpips_loss

    def log_metrics(self, prefix, loss, mse, lpips, ssim=None):
        values = {
            f'{prefix}/loss': loss,
            f'{prefix}/mse': mse,
            f'{prefix}/psnr': -10.0 * mse.log10(),
            f'{prefix}/lpips': lpips,
        }

        if ssim is not None:
            values[f'{prefix}/ssim'] = ssim

        prog_bar = prefix != 'val'
        sync_dist = prefix != 'train'
        self.log_dict(values, prog_bar=prog_bar, sync_dist=sync_dist, batch_size=self.config.data.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.config.opt.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [self.config.opt.epochs // 2], gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
    
    def calculate_overlap(self, view1, view2, target, is_optical=False):
        # print(target["original_img"].shape)
        # print(view1["original_img"].shape)
        border = random.randint(100, 150)
        b, _, h, w = target["original_img"].shape
        if is_optical:
            with torch.no_grad():
                # flow_1 = self.raft(target["original_img"] * 2 - 1, view1["original_img"] * 2 - 1)[-1].square() * torch.logical_and(target["maskmap"], view1["maskmap"])
                flow_1 = self.raft(view1["original_img"] * 2 - 1, target["original_img"] * 2 - 1)[-1].square() * torch.logical_and(target["maskmap"], view1["maskmap"]).unsqueeze(1).int()
                diff_1 = torch.sqrt(flow_1[:, 0, :, :] + flow_1[:, 1, :, :])
                # flow_2 = self.raft(target["original_img"] * 2 - 1, view2["original_img"] * 2 - 1)[-1].square() * torch.logical_and(target["maskmap"], view2["maskmap"])
                flow_2 = self.raft(view2["original_img"] * 2 - 1, target["original_img"] * 2 - 1)[-1].square() * torch.logical_and(target["maskmap"], view2["maskmap"]).unsqueeze(1).int()
                diff_2 = torch.sqrt(flow_2[:, 0, :, :] + flow_2[:, 1, :, :])
                flow_input = self.raft(view1["original_img"] * 2 - 1, view2["original_img"] * 2 - 1)[-1].square() * torch.logical_and(view1["maskmap"], view2["maskmap"]).unsqueeze(1).int()
                diff_input = torch.sqrt(flow_input[:, 0, :, :] + flow_input[:, 1, :, :])
        else:   
            diff_1 = (target["original_img"] - view1["original_img"]) 
            diff_1 = torch.abs(diff_1[:, 0, :, :]) + torch.abs(diff_1[:, 1, :, :]) + torch.abs(diff_1[:, 2, :, :])
            diff_2 = (target["original_img"] - view2["original_img"])
            diff_2 = torch.abs(diff_2[:, 0, :, :]) + torch.abs(diff_2[:, 1, :, :]) + torch.abs(diff_2[:, 2, :, :])
        mask1 = torch.ones_like(diff_1)
        mask2 = torch.ones_like(diff_2)
        # mask_b = torch.ones_like(diff_1)
        # mask_b[:, :border, :] = 0
        # mask_b[:, :, :border] = 0
        # mask_b[:, h - border:, :] = 0
        # mask_b[:, :, w - border :] = 0
        # print(torch.quantile(diff_1.reshape(b, h*w), 0.7, dim=1, keepdim=True).shape)
        # diff_1_norm = diff_1/(torch.max(diff_1.reshape(b, h*w), dim=1, keepdim=True)[0].unsqueeze(1)*2)
        # diff_2_norm = diff_2/(torch.max(diff_2.reshape(b, h*w), dim=1, keepdim=True)[0].unsqueeze(1)*2)
        diff_input_norm = diff_input/(torch.max(diff_input.reshape(b, h*w), dim=1, keepdim=True)[0].unsqueeze(1))
        diff_input[diff_input_norm<=0.2] = 1
        diff_input[diff_input_norm>0.2] = 0
        # diff_input = 1 - diff_input
        
        mask_all = diff_input * torch.rand_like(diff_1)
        # mask1[diff_1>diff_2] = 0
        mask1[mask_all>0.5] = 0
        # mask2[diff_2>diff_1] = 0
        mask2[torch.logical_and(mask_all>0, mask_all<=0.5)] = 0
        # mask1[diff_1>torch.quantile(diff_1.reshape(b, h*w), 0.8, dim=1, keepdim=True).unsqueeze(1)] = 0
        # mask2[diff_2>torch.quantile(diff_2.reshape(b, h*w), 0.8, dim=1, keepdim=True).unsqueeze(1)] = 0
        # print(torch.sum(mask1) / 8)
        # mask = torch.logical_and(torch.logical_or(mask1, mask2), torch.logical_and(torch.logical_or(view1["maskmap"], view2["maskmap"]), target["maskmap"])).int()
        
        mask_0 = torch.minimum(diff_1, diff_2)
        mask_0 = mask_0/(torch.max(mask_0.reshape(b, h*w), dim=1, keepdim=True)[0].unsqueeze(1)*2)
        mask = (1-mask_0)
        mask[mask_0==0] = 0
        mask = mask * torch.logical_and(torch.logical_or(view1["maskmap"], view2["maskmap"]), target["maskmap"]).int() * 0.50 #* mask_b
        return torch.logical_and(mask1, view1["maskmap"]).int(), torch.logical_and(mask2, view2["maskmap"]).int(), mask.unsqueeze(1)

def run_experiment(config):

    # Set the seed
    L.seed_everything(config.seed, workers=True)

    # Set up loggers
    os.makedirs(os.path.join(config.save_dir, config.name), exist_ok=True)
    loggers = []
    if config.loggers.use_csv_logger:
        csv_logger = L.pytorch.loggers.CSVLogger(
            save_dir=config.save_dir,
            name=config.name
        )
        loggers.append(csv_logger)
    if config.loggers.use_wandb:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            project='splatt3r',
            name=config.name,
            save_dir=config.save_dir,
            config=omegaconf.OmegaConf.to_container(config),
        )
        if wandb.run is not None:
            wandb.run.log_code(".")
        loggers.append(wandb_logger)

    # Set up profiler
    if config.use_profiler:
        profiler = L.pytorch.profilers.PyTorchProfiler(
            dirpath=config.save_dir,
            filename='trace',
            export_to_chrome=True,
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=3),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(config.save_dir),
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            profile_memory=True,
            with_stack=True
        )
    else:
        profiler = None

    # Model
    print('Loading Model')
    model = MAST3RGaussians(config)
    if config.use_pretrained:
        ckpt = torch.load(config.pretrained_mast3r_path)
        # _ = model.encoder.load_state_dict(ckpt['model'], strict=False)
        # print(ckpt["state_dict"].keys())
        # train with splatt3r pretrain weights
        _ = model.load_state_dict(ckpt["state_dict"], strict=False)
        del ckpt

    # Training Datasets
    print(f'Building Datasets')
    train_dataset = endonerf.get_endo_dataset(
        config.data.root,
        'train',
        config.data.resolution,
        num_epochs_per_epoch=config.data.epochs_per_train_epoch,
    )
    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )

    val_dataset = endonerf.get_endo_test_dataset(
        config.data.root,
        alpha=0.5,
        beta=0.5,
        resolution=config.data.resolution,
        use_every_n_sample=100,
    )
    data_loader_val = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )

    # Training
    print('Training')
    trainer = L.Trainer(
        accelerator="gpu", 
        benchmark=True,
        callbacks=[
            L.pytorch.callbacks.LearningRateMonitor(logging_interval='epoch', log_momentum=True),
            export.SaveBatchData(save_dir=config.save_dir),
        ],
        check_val_every_n_epoch=100,
        default_root_dir=config.save_dir,
        devices=config.devices,
        gradient_clip_val=config.opt.gradient_clip_val,
        log_every_n_steps=10,
        logger=loggers,
        max_epochs=config.opt.epochs,
        profiler=profiler,
        strategy="ddp_find_unused_parameters_true" if len(config.devices) > 1 else "auto",
    )
    trainer.fit(model, train_dataloaders=data_loader_train, val_dataloaders=data_loader_val)
    torch.save(model.encoder.downstream_head1.gaussian_dpt.dpt.state_dict(), os.path.join(config.save_dir, "endo_retrained_head1_gaussain_dpt.pth"))
    torch.save(model.encoder.downstream_head2.gaussian_dpt.dpt.state_dict(), os.path.join(config.save_dir, "endo_retrained_head2_gaussain_dpt.pth"))
    # torch.save(model.encoder.downstream_head1.state_dict(), os.path.join(config.save_dir, "endo_retrained_head1.pth"))
    # torch.save(model.encoder.downstream_head2.state_dict(), os.path.join(config.save_dir, "endo_retrained_head2.pth"))
    
    # exit()
    
    # Testing
    original_save_dir = config.save_dir
    results = {}
    for alpha, beta in ((0.9, 0.9),):

        test_dataset = endonerf.get_endo_test_dataset(
            config.data.root,
            alpha=alpha,
            beta=beta,
            resolution=config.data.resolution,
            use_every_n_sample=10
        )
        data_loader_test = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
        )

        masking_configs = ((True, False), (True, True))
        for apply_mask, average_over_mask in masking_configs:

            new_save_dir = os.path.join(
                original_save_dir,
                f'alpha_{alpha}_beta_{beta}_apply_mask_{apply_mask}_average_over_mask_{average_over_mask}'
            )
            os.makedirs(new_save_dir, exist_ok=True)
            model.config.save_dir = new_save_dir

            L.seed_everything(config.seed, workers=True)

            # Training
            trainer = L.Trainer(
                accelerator="gpu",
                benchmark=True,
                callbacks=[export.SaveBatchData(save_dir=config.save_dir),],
                default_root_dir=config.save_dir,
                devices=config.devices,
                log_every_n_steps=10,
                strategy="ddp_find_unused_parameters_true" if len(config.devices) > 1 else "auto",
            )

            model.lpips_criterion = lpips.LPIPS('vgg', spatial=average_over_mask)
            model.config.loss.apply_mask = apply_mask
            model.config.loss.average_over_mask = average_over_mask
            res = trainer.test(model, dataloaders=data_loader_test)
            results[f"alpha: {alpha}, beta: {beta}, apply_mask: {apply_mask}, average_over_mask: {average_over_mask}"] = res

            # Save the results
            save_path = os.path.join(original_save_dir, 'results.json')
            with open(save_path, 'w') as f:
                json.dump(results, f)


if __name__ == "__main__":

    # Setup the workspace (eg. load the config, create a directory for results at config.save_dir, etc.)
    config = workspace.load_config(sys.argv[1], sys.argv[2:])
    if os.getenv("LOCAL_RANK", '0') == '0':
        config = workspace.create_workspace(config)

    # Run training
    run_experiment(config)
