import os

from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation
import einops
import numpy as np
import torch
import torchvision
import trimesh
import lightning as L
import open3d as o3d

import utils.loss_mask as loss_mask
from src.mast3r_src.dust3r.dust3r.viz import OPENGL, pts3d_to_trimesh, cat_meshes


C0 = 0.28209479177387814

class SaveBatchData(L.Callback):
    '''A Lightning callback that occasionally saves batch inputs and outputs to disk.
    It is not critical to the training process, and can be disabled if unwanted.'''

    def __init__(self, save_dir, train_save_interval=100, val_save_interval=100, test_save_interval=100):
        self.save_dir = save_dir
        self.train_save_interval = train_save_interval
        self.val_save_interval = val_save_interval
        self.test_save_interval = test_save_interval

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.train_save_interval == 0 and trainer.global_rank == 0:
            self.save_batch_data('train', trainer, pl_module, batch, batch_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.val_save_interval == 0 and trainer.global_rank == 0:
            self.save_batch_data('val', trainer, pl_module, batch, batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.test_save_interval == 0 and trainer.global_rank == 0:
            self.save_batch_data('test', trainer, pl_module, batch, batch_idx)

    def save_batch_data(self, prefix, trainer, pl_module, batch, batch_idx):

        print(f'Saving {prefix} data at epoch {trainer.current_epoch} and batch {batch_idx}')

        # Run the batch through the model again
        _, _, h, w = batch["context"][0]["img"].shape
        view1, view2 = batch['context']
        pred1, pred2 = pl_module.forward(view1, view2)
        color, depth = pl_module.decoder(batch, pred1, pred2, (h, w))
        # mask = loss_mask.calculate_loss_mask(batch)
        mask = torch.stack([target_view['maskmap'] for target_view in batch['target']], dim=1)

        # Save the data
        save_dir = os.path.join(
            self.save_dir,
            f"{prefix}_epoch_{trainer.current_epoch}_batch_{batch_idx}"
        )
        log_batch_files(batch, color, depth, mask, view1, view2, pred1, pred2, save_dir)


def save_as_ply(pred1, pred2, save_path):
    """Save the 3D Gaussians as a point cloud in the PLY format.
    Adapted loosely from PixelSplat"""

    def construct_list_of_attributes(num_rest: int) -> list[str]:
        '''Construct a list of attributes for the PLY file format. This
        corresponds to the attributes used by online readers, such as
        https://niujinshuchong.github.io/mip-splatting-demo/index.html'''
        attributes = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(3):
            attributes.append(f"f_dc_{i}")
        for i in range(num_rest):
            attributes.append(f"f_rest_{i}")
        attributes.append("opacity")
        for i in range(3):
            attributes.append(f"scale_{i}")
        for i in range(4):
            attributes.append(f"rot_{i}")
        return attributes

    def covariance_to_quaternion_and_scale(covariance):
        '''Convert the covariance matrix to a four dimensional quaternion and
        a three dimensional scale vector'''

        # Perform singular value decomposition
        U, S, V = torch.linalg.svd(covariance)

        # The scale factors are the square roots of the eigenvalues
        scale = torch.sqrt(S)
        scale = scale.detach().cpu().numpy()

        # The rotation matrix is U*Vt
        rotation_matrix = torch.bmm(U, V.transpose(-2, -1))
        rotation_matrix_np = rotation_matrix.detach().cpu().numpy()

        # Use scipy to convert the rotation matrix to a quaternion
        rotation = Rotation.from_matrix(rotation_matrix_np)
        quaternion = rotation.as_quat()

        return quaternion, scale
    
    def change_ply_format(plydata, gs_path):
        
        def SH2RGB(sh):
            return sh * C0 + 0.5
  
        # plydata = PlyData.read(gs_path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        rgb = SH2RGB(features_dc[..., 0])
        # clamp the lower bound of rgb values to 0
        rgb = np.maximum(rgb, 0)

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        opacities = torch.sigmoid(torch.from_numpy(opacities)).numpy()
        opacity_mask = (opacities > 0.005).squeeze(1)
        xyz = xyz[opacity_mask]
        rgb = rgb[opacity_mask]

        # for point with rgb values large than 1, we need to rescale all channels by making the largest channel 1
        max_rgb = np.max(rgb, axis=1)
        max_rgb = np.maximum(max_rgb, 1)
        rgb = rgb / max_rgb[:, np.newaxis]

        # for checking
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        o3d.io.write_point_cloud(gs_path.replace("gaussians.ply", "colored.ply"), pcd)

    # Collect the Gaussian parameters
    means = torch.stack([pred1["means"], pred2["means_in_other_view"]], dim=1)
    covariances = torch.stack([pred1["covariances"], pred2["covariances"]], dim=1)
    harmonics = torch.stack([pred1["sh"], pred2["sh"]], dim=1)[..., 0]  # Only use the first harmonic
    opacities = torch.stack([pred1["opacities"], pred2["opacities"]], dim=1)

    # Rearrange the tensors to the correct shape
    means = einops.rearrange(means[0], "view h w xyz -> (view h w) xyz").detach().cpu().numpy()
    covariances = einops.rearrange(covariances[0], "v h w i j -> (v h w) i j")
    harmonics = einops.rearrange(harmonics[0], "view h w xyz -> (view h w) xyz").detach().cpu().numpy()
    opacities = einops.rearrange(opacities[0], "view h w xyz -> (view h w) xyz").detach().cpu().numpy()

    # Convert the covariance matrices to quaternions and scales
    rotations, scales = covariance_to_quaternion_and_scale(covariances)

    # Construct the attributes
    rest = np.zeros_like(means)
    attributes = np.concatenate((means, rest, harmonics, opacities, np.log(scales), rotations), axis=-1)
    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0)]
    elements = np.empty(attributes.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))

    # Save the point cloud
    point_cloud = PlyElement.describe(elements, "vertex")
    scene = PlyData([point_cloud])
    scene.write(save_path)
    change_ply_format(scene, save_path)


def save_3d(view1, view2, pred1, pred2, save_dir, as_pointcloud=True, all_points=True):
    """Save the 3D points as a point cloud or as a mesh. Adapted from DUSt3R"""

    os.makedirs(save_dir, exist_ok=True)
    batch_size = pred1["pts3d"].shape[0]
    views = [view1, view2]

    for b in range(batch_size):

        pts3d = [pred1["pts3d"][b].cpu().numpy()] + [pred2["pts3d_in_other_view"][b].cpu().numpy()]
        imgs = [einops.rearrange(view["original_img"][b], "c h w -> h w c").cpu().numpy() for view in views]
        mask = [view["valid_mask"][b].cpu().numpy() for view in views]

        # Treat all pixels as valid, because we want to render the entire viewpoint
        if all_points:
            mask = [np.ones_like(m) for m in mask]

        # Construct the scene from the 3D points as a point cloud or as a mesh
        scene = trimesh.Scene()
        if as_pointcloud:
            pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
            col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
            pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
            scene.add_geometry(pct)
            save_path = os.path.join(save_dir, f"{b}.ply")
        else:
            meshes = []
            for i in range(len(imgs)):
                meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
            mesh = trimesh.Trimesh(**cat_meshes(meshes))
            scene.add_geometry(mesh)
            save_path = os.path.join(save_dir, f"{b}.glb")

        # Save the scene
        scene.export(file_obj=save_path)


@torch.no_grad()
def log_batch_files(batch, color, depth, mask, view1, view2, pred1, pred2, save_dir, should_save_3d=False):
    '''Save all the relevant debug files for a batch'''

    os.makedirs(save_dir, exist_ok=True)

    # Save the 3D Gaussians as a .ply file
    save_as_ply(pred1, pred2, os.path.join(save_dir, f"gaussians.ply"))

    # Save the 3D points as a point cloud and as a mesh (disabled)
    if should_save_3d:
        save_3d(view1, view2, pred1, pred2, os.path.join(save_dir, "3d_mesh"), as_pointcloud=False)
        save_3d(view1, view2, pred1, pred2, os.path.join(save_dir, "3d_pointcloud"), as_pointcloud=True)

    # Save the color, depth and valid masks for the input context images
    context_images = torch.stack([view["img"] for view in batch["context"]], dim=1)
    context_original_images = torch.stack([view["original_img"] for view in batch["context"]], dim=1)
    context_depthmaps = torch.stack([view["maskmap"] for view in batch["context"]], dim=1)
    # context_valid_masks = torch.stack([view["valid_mask"] for view in batch["context"]], dim=1)
    for b in range(min(context_images.shape[0], 10)):
        torchvision.utils.save_image(context_images[b], os.path.join(save_dir, f"sample_{b}_img_context.jpg"))
        torchvision.utils.save_image(context_original_images[b], os.path.join(save_dir, f"sample_{b}_original_img_context.jpg"))
        # torchvision.utils.save_image(context_depthmaps[b, :, None, ...], os.path.join(save_dir, f"sample_{b}_depthmap.jpg"))#, normalize=True)
        # torchvision.utils.save_image(context_valid_masks[b, :, None, ...].float(), os.path.join(save_dir, f"sample_{b}_valid_mask_context.jpg"), normalize=True)

    # Save the color and depth images for the target images
    target_original_images = torch.stack([view["original_img"] for view in batch["target"]], dim=1)
    target_depthmaps = torch.stack([view["maskmap"] for view in batch["target"]], dim=1)
    # context_valid_masks = torch.stack([view["valid_mask"] for view in batch["context"]], dim=1)
    for b in range(min(target_original_images.shape[0], 10)):
        torchvision.utils.save_image(target_original_images[b], os.path.join(save_dir, f"sample_{b}_original_img_target.jpg"))
        # torchvision.utils.save_image(target_depthmaps[b, :, None, ...], os.path.join(save_dir, f"sample_{b}_depthmap_target.jpg"))#, normalize=True)

    # Save the rendered images and depths
    for b in range(min(color.shape[0], 10)):
        torchvision.utils.save_image(color[b, ...], os.path.join(save_dir, f"sample_{b}_rendered_color.jpg"))
    # if depth is not None:
    #     for b in range(min(color.shape[0], 4)):
            # torchvision.utils.save_image(depth[b, :, None, ...], os.path.join(save_dir, f"sample_{b}_rendered_depth.jpg"))#, normalize=True)

    # Save the loss masks
    for b in range(min(mask.shape[0], 10)):
        torchvision.utils.save_image(mask[b, :, None, ...].float(), os.path.join(save_dir, f"sample_{b}_loss_mask.jpg"), normalize=True)

    # Save the masked target and rendered images
    target_original_images = torch.stack([view["original_img"] for view in batch["target"]], dim=1)
    masked_target_original_images = target_original_images * mask[..., None, :, :]
    masked_predictions = color * mask[..., None, :, :]
    for b in range(min(target_original_images.shape[0], 10)):
        torchvision.utils.save_image(masked_target_original_images[b], os.path.join(save_dir, f"sample_{b}_masked_original_img_target.jpg"))
        torchvision.utils.save_image(masked_predictions[b], os.path.join(save_dir, f"sample_{b}_masked_rendered_color.jpg"))
        
def save_rgb_ply(pred1: dict, save_path):
    """Save the 3D Gaussians as a point cloud in the PLY format.
    Adapted loosely from PixelSplat"""

    def construct_list_of_attributes(num_rest: int) -> list[str]:
        '''Construct a list of attributes for the PLY file format. This
        corresponds to the attributes used by online readers, such as
        https://niujinshuchong.github.io/mip-splatting-demo/index.html'''
        attributes = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(3):
            attributes.append(f"f_dc_{i}")
        for i in range(num_rest):
            attributes.append(f"f_rest_{i}")
        attributes.append("opacity")
        for i in range(3):
            attributes.append(f"scale_{i}")
        for i in range(4):
            attributes.append(f"rot_{i}")
        return attributes

    def covariance_to_quaternion_and_scale(covariance):
        '''Convert the covariance matrix to a four dimensional quaternion and
        a three dimensional scale vector'''

        # Perform singular value decomposition
        U, S, V = torch.linalg.svd(covariance)

        # The scale factors are the square roots of the eigenvalues
        scale = torch.sqrt(S)
        scale = scale.detach().cpu().numpy()

        # The rotation matrix is U*Vt
        rotation_matrix = torch.bmm(U, V.transpose(-2, -1))
        rotation_matrix_np = rotation_matrix.detach().cpu().numpy()

        # Use scipy to convert the rotation matrix to a quaternion
        rotation = Rotation.from_matrix(rotation_matrix_np)
        quaternion = rotation.as_quat()

        return quaternion, scale
    
    def change_ply_format(plydata, gs_path):
        
        def SH2RGB(sh):
            return sh * C0 + 0.5
  
        # plydata = PlyData.read(gs_path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        rgb = SH2RGB(features_dc[..., 0])
        # clamp the lower bound of rgb values to 0
        rgb = np.maximum(rgb, 0)

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        opacities = torch.sigmoid(torch.from_numpy(opacities)).numpy()
        opacity_mask = (opacities > 0.005).squeeze(1)
        xyz = xyz[opacity_mask]
        rgb = rgb[opacity_mask]

        # for point with rgb values large than 1, we need to rescale all channels by making the largest channel 1
        max_rgb = np.max(rgb, axis=1)
        max_rgb = np.maximum(max_rgb, 1)
        rgb = rgb / max_rgb[:, np.newaxis]

        # for checking
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        o3d.io.write_point_cloud(gs_path.replace("gaussians.ply", "colored.ply"), pcd)

    # Collect the Gaussian parameters
    
    means = pred1.get("means", 0) if "means" in pred1.keys() else pred1.get("means_in_other_view", 0)
    covariances = pred1["covariances"]
    harmonics = pred1["sh"][..., 0]  # Only use the first harmonic
    opacities = pred1["opacities"]

    # Rearrange the tensors to the correct shape
    means = einops.rearrange(means[0], "h w xyz -> (h w) xyz").detach().cpu().numpy()
    covariances = einops.rearrange(covariances[0], "h w i j -> (h w) i j")
    harmonics = einops.rearrange(harmonics[0], "h w xyz -> (h w) xyz").detach().cpu().numpy()
    opacities = einops.rearrange(opacities[0], "h w xyz -> (h w) xyz").detach().cpu().numpy()

    # Convert the covariance matrices to quaternions and scales
    rotations, scales = covariance_to_quaternion_and_scale(covariances)

    # Construct the attributes
    rest = np.zeros_like(means)
    attributes = np.concatenate((means, rest, harmonics, opacities, np.log(scales), rotations), axis=-1)
    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0)]
    elements = np.empty(attributes.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))

    # Save the point cloud
    point_cloud = PlyElement.describe(elements, "vertex")
    scene = PlyData([point_cloud])
    scene.write(save_path)
    change_ply_format(scene, save_path)