#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# NOTE: This code modifies 3DGS with the support for cx, cy not in the center of the image
# It also adds support for sampling masks

import dataclasses
import math
import warnings
import random
import itertools
import shlex
import logging
import copy
from typing import Optional
import os
import tempfile
import numpy as np
from PIL import Image
from nerfbaselines import (
    Method, MethodInfo, ModelInfo, RenderOutput, Cameras, camera_model_to_int, Dataset
)
import shlex

from argparse import ArgumentParser

import torch
from random import randint

from scipy.cluster.hierarchy import single

from utils.general_utils import PILtoTorch  # type: ignore
from arguments import ModelParams, PipelineParams, OptimizationParams #  type: ignore
from gaussian_renderer import render # type: ignore
from scene import GaussianModel # type: ignore
import scene.dataset_readers  # type: ignore
from scene.dataset_readers import SceneInfo, getNerfppNorm, focal2fov  # type: ignore
from scene.dataset_readers import CameraInfo as _old_CameraInfo  # type: ignore
from scene.dataset_readers import storePly, fetchPly  # type: ignore
from utils.general_utils import safe_state  # type: ignore
from utils.graphics_utils import fov2focal, getWorld2View2  # type: ignore
from utils.loss_utils import l1_loss, ssim  # type: ignore
from utils.sh_utils import SH2RGB, eval_sh  # type: ignore
from scene import Scene, sceneLoadTypeCallbacks  # type: ignore
from utils import camera_utils  # type: ignore

from encoders import AppearanceTransform, initialize_weights

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision.utils import save_image


def convert_image_dtype(image: np.ndarray, dtype) -> np.ndarray:
    if image.dtype == dtype:
        return image
    if image.dtype != np.uint8 and dtype != np.uint8:
        return image.astype(dtype)
    if image.dtype == np.uint8 and dtype != np.uint8:
        return image.astype(dtype) / 255.0
    if image.dtype != np.uint8 and dtype == np.uint8:
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)
    raise ValueError(f"cannot convert image from {image.dtype} to {dtype}")


def scale_grads(values, scale):
    grad_values = values * scale
    rest_values = values.detach() * (1 - scale)
    return grad_values + rest_values


def flatten_hparams(hparams, *, separator: str = "/", _prefix: str = ""):
    flat = {}
    if dataclasses.is_dataclass(hparams):
        hparams = {f.name: getattr(hparams, f.name) for f in dataclasses.fields(hparams)}
    for k, v in hparams.items():
        if _prefix:
            k = f"{_prefix}{separator}{k}"
        if isinstance(v, dict) or dataclasses.is_dataclass(v):
            flat.update(flatten_hparams(v, _prefix=k, separator=separator).items())
        else:
            flat[k] = v
    return flat


def getProjectionMatrixFromOpenCV(w, h, fx, fy, cx, cy, znear, zfar):
    z_sign = 1.0
    P = torch.zeros((4, 4))
    P[0, 0] = 2.0 * fx / w
    P[1, 1] = 2.0 * fy / h
    P[0, 2] = (2.0 * cx - w) / w
    P[1, 2] = (2.0 * cy - h) / h
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


#
# Patch Gaussian Splatting to include sampling masks
# Also, fix cx, cy (ignored in gaussian-splatting)
#
# Patch loadCam to include sampling mask
_old_loadCam = camera_utils.loadCam
def loadCam(args, id, cam_info, resolution_scale):
    camera = _old_loadCam(args, id, cam_info, resolution_scale)

    sampling_mask = None
    if cam_info.sampling_mask is not None:
        sampling_mask = PILtoTorch(cam_info.sampling_mask, (camera.image_width, camera.image_height))
    setattr(camera, "sampling_mask", sampling_mask)
    setattr(camera, "_patched", True)

    # Fix cx, cy (ignored in gaussian-splatting)
    camera.focal_x = fov2focal(cam_info.FovX, camera.image_width)
    camera.focal_y = fov2focal(cam_info.FovY, camera.image_height)
    camera.cx = cam_info.cx
    camera.cy = cam_info.cy
    camera.projection_matrix = getProjectionMatrixFromOpenCV(
        camera.image_width,
        camera.image_height,
        camera.focal_x,
        camera.focal_y,
        camera.cx,
        camera.cy,
        camera.znear,
        camera.zfar).transpose(0, 1).cuda()
    camera.full_proj_transform = (camera.world_view_transform.unsqueeze(0).bmm(camera.projection_matrix.unsqueeze(0))).squeeze(0)

    return camera
camera_utils.loadCam = loadCam


# Patch CameraInfo to add sampling mask
class CameraInfo(_old_CameraInfo):
    def __new__(cls, *args, sampling_mask=None, cx, cy, **kwargs):
        self = super(CameraInfo, cls).__new__(cls, *args, **kwargs)
        self.sampling_mask = sampling_mask
        self.cx = cx
        self.cy = cy
        return self
scene.dataset_readers.CameraInfo = CameraInfo


def _load_caminfo(idx, pose, intrinsics, image_name, image_size, image=None, image_path=None, sampling_mask=None, scale_coords=None):
    pose = np.copy(pose)
    pose = np.concatenate([pose, np.array([[0, 0, 0, 1]], dtype=pose.dtype)], axis=0)
    pose = np.linalg.inv(pose)
    R = pose[:3, :3]
    T = pose[:3, 3]
    if scale_coords is not None:
        T = T * scale_coords
    R = np.transpose(R)

    width, height = image_size
    fx, fy, cx, cy = intrinsics
    if image is None:
        image = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
    return CameraInfo(
        uid=idx, R=R, T=T,
        FovX=focal2fov(float(fx), float(width)),
        FovY=focal2fov(float(fy), float(height)),
        image=image, image_path=image_path, image_name=image_name,
        width=int(width), height=int(height),
        sampling_mask=sampling_mask,
        cx=cx, cy=cy)


def _config_overrides_to_args_list(args_list, config_overrides):
    for k, v in config_overrides.items():
        if str(v).lower() == "true":
            v = True
        if str(v).lower() == "false":
            v = False
        if isinstance(v, bool):
            if v:
                if f'--{k}' not in args_list:
                    args_list.append(f'--{k}')
            else:
                if f'--{k}' in args_list:
                    args_list.remove(f'--{k}')
        elif f'--{k}' in args_list:
            args_list[args_list.index(f'--{k}') + 1] = str(v)
        else:
            args_list.append(f"--{k}")
            args_list.append(str(v))


def _convert_dataset_to_gaussian_splatting(dataset: Optional[Dataset], tempdir: str, white_background: bool = False, scale_coords=None):
    if dataset is None:
        return SceneInfo(None, [], [], nerf_normalization=dict(radius=None, translate=None), ply_path=None)
    assert np.all(dataset["cameras"].camera_models == camera_model_to_int("pinhole")), "Only pinhole cameras supported"

    cam_infos = []
    for idx, extr in enumerate(dataset["cameras"].poses):
        del extr
        intrinsics = dataset["cameras"].intrinsics[idx]
        pose = dataset["cameras"].poses[idx]
        image_path = dataset["image_paths"][idx] if dataset["image_paths"] is not None else f"{idx:06d}.png"
        image_name = (
            os.path.relpath(str(dataset["image_paths"][idx]), str(dataset["image_paths_root"])) if dataset["image_paths"] is not None and dataset["image_paths_root"] is not None else os.path.basename(image_path)
        )

        w, h = dataset["cameras"].image_sizes[idx]
        im_data = dataset["images"][idx][:h, :w]
        assert im_data.dtype == np.uint8, "Gaussian Splatting supports images as uint8"
        if im_data.shape[-1] == 4:
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + (1 - norm_data[:, :, 3:4]) * bg
            im_data = np.array(arr * 255.0, dtype=np.uint8)
        if not white_background and dataset["metadata"].get("id") == "blender":
            warnings.warn("Blender scenes are expected to have white background. If the background is not white, please set white_background=True in the dataset loader.")
        elif white_background and dataset["metadata"].get("id") != "blender":
            warnings.warn("white_background=True is set, but the dataset is not a blender scene. The background may not be white.")
        image = Image.fromarray(im_data)
        sampling_mask = None
        if dataset["sampling_masks"] is not None:
            sampling_mask = Image.fromarray((dataset["sampling_masks"][idx] * 255).astype(np.uint8))

        cam_info = _load_caminfo(
            idx, pose, intrinsics,
            image_name=image_name,
            image_path=image_path,
            image_size=(w, h),
            image=image,
            sampling_mask=sampling_mask,
            scale_coords=scale_coords,
        )
        cam_infos.append(cam_info)

    cam_infos = sorted(cam_infos.copy(), key=lambda x: x.image_name)
    nerf_normalization = getNerfppNorm(cam_infos)

    points3D_xyz = dataset["points3D_xyz"]
    if scale_coords is not None:
        points3D_xyz = points3D_xyz * scale_coords
    points3D_rgb = dataset["points3D_rgb"]
    if points3D_xyz is None and dataset["metadata"].get("id", None) == "blender":
        # https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/dataset_readers.py#L221C4-L221C4
        num_pts = 100_000
        logging.info(f"generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        points3D_xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        points3D_rgb = (SH2RGB(shs) * 255).astype(np.uint8)

    storePly(os.path.join(tempdir, "scene.ply"), points3D_xyz, points3D_rgb)
    pcd = fetchPly(os.path.join(tempdir, "scene.ply"))
    scene_info = SceneInfo(point_cloud=pcd, train_cameras=cam_infos, test_cameras=[], nerf_normalization=nerf_normalization, ply_path=os.path.join(tempdir, "scene.ply"))
    return scene_info


def _sample_view(viewpoint_stack, correspond_scene):
    # Pick a random Camera
    if not viewpoint_stack:
        loadCam.was_called = False  # type: ignore
        viewpoint_stack = correspond_scene.getTrainCameras().copy()
        if any(not getattr(cam, "_patched", False) for cam in viewpoint_stack):
            raise RuntimeError("could not patch loadCam!")
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
    return viewpoint_cam


def _pre_iteratrion(iteration, gaussians, viewpoint_stack, scene, global_encoding_list):
    # Update learning rate
    gaussians.update_learning_rate(iteration)

    # Every 1000 its we increase the levels of SH up to a maximum degree
    if iteration % 1000 == 0:
        gaussians.oneupSHdegree()

    # sample viewpoint
    viewpoint_cam = _sample_view(viewpoint_stack, scene)
    # todo check if using viewpoint_cam.uid is correct? Fix during entire training
    global_encoding = global_encoding_list[viewpoint_cam.uid]

    return viewpoint_cam, global_encoding


def _save_heat_map(image, name, iteration):
    image_uint8 = (image * 255).astype(np.uint8)
    save_root = "/home/lorentz/Project/Code/gaussian-splatting/testing_w3dgs_bgate_vis_residual/residual"
    Image.fromarray(image_uint8).save(os.path.join(save_root, f"{iteration}_{name}.png"))


class GaussianSplatting(Method):
    def __init__(self, *,
                 checkpoint: Optional[str] = None,
                 train_dataset: Optional[Dataset] = None,
                 config_overrides: Optional[dict] = None):
        self.checkpoint = checkpoint
        self.background = None
        self.step = 0

        # Setup parameters
        self._args_list = ["--source_path", "<empty>", "--resolution", "1", "--eval"]
        self._loaded_step = None
        if checkpoint is not None:
            if not os.path.exists(checkpoint):
                raise RuntimeError(f"Model directory {checkpoint} does not exist")
            with open(os.path.join(checkpoint, "args.txt"), "r", encoding="utf8") as f:
                self._args_list = shlex.split(f.read())
            self._loaded_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(str(checkpoint)) if x.startswith("chkpnt-"))[-1]

        # Fix old checkpoints
        if "--resolution" not in self._args_list:
            self._args_list.extend(("--resolution", "1"))

        if self.checkpoint is None and config_overrides is not None:
            _config_overrides_to_args_list(self._args_list, config_overrides)

        self._load_config()

        if self.checkpoint is None:
            # Verify parameters are set correctly
            assert train_dataset is not None, "train_dataset must be set if checkpoint is not provided"
            if train_dataset["metadata"].get("id") == "blender" and not self.dataset.white_background:
                warnings.warn("white_background should be True for blender dataset")

        self._setup(train_dataset)

    def _load_config(self):
        parser = ArgumentParser(description="Training script parameters")
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)
        parser.add_argument("--scale_coords", type=float, default=None, help="Scale the coords")
        args = parser.parse_args(self._args_list)
        self.dataset = lp.extract(args)
        self.dataset.scale_coords = args.scale_coords
        self.opt = op.extract(args)
        self.pipe = pp.extract(args)

    def _setup(self, train_dataset):
        # Initialize system state (RNG)
        safe_state(False)

        # Setup model
        self.gaussians_1 = GaussianModel(self.dataset.sh_degree)
        self.gaussians_2 = GaussianModel(self.dataset.sh_degree)
        self.scene_1 = self._build_scene(train_dataset, self.gaussians_1)
        self.scene_2 = self._build_scene(train_dataset, self.gaussians_2)
        if train_dataset is not None:
            self.gaussians_1.training_setup(self.opt)
            self.gaussians_2.training_setup(self.opt)
        # todo later load weight
        # if train_dataset is None or self.checkpoint:
        #     info = self.get_info()
        #     loaded_step = info.get("loaded_step")
        #     assert loaded_step is not None, "Could not infer loaded step"
        #     (model_params, self.step) = torch.load(str(self.checkpoint) + f"/chkpnt-{loaded_step}.pth",
        #                                            weights_only=False)
        #     self.gaussians.restore(model_params, self.opt)

        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self._viewpoint_stack_1 = []
        self._viewpoint_stack_2 = []

        # self._input_points = None
        # if train_dataset is not None:
        #     self._input_points = (train_dataset["points3D_xyz"], train_dataset["points3D_rgb"])

        # todo tuning these models
        encoding_dim = 32
        appearance_n_fourier_freqs = 4
        # self.appearance_encoder = AppearanceEncoder(backbone="resnet18", output_dim=encoding_dim, pretrained=False).cuda()
        # self.appearance_encoder = AppearanceEncoder(output_dim=encoding_dim).cuda()
        # self.appearance_encoder.apply(initialize_weights)
        self.appearance_transform = AppearanceTransform(global_encoding_dim=encoding_dim, local_encoding_dim=appearance_n_fourier_freqs*6).cuda()
        self.appearance_transform.apply(initialize_weights)
        # self.appearance_encoder_optimizer = torch.optim.Adam(self.appearance_encoder.parameters(), lr=0.0005, eps=1e-15)
        self.appearance_transform_optimizer = torch.optim.Adam(self.appearance_transform.parameters(), lr=0.0005, eps=1e-15)

        self.global_encoding_1 = torch.normal(mean=0, std=0.01, size=(len(train_dataset["images"]), encoding_dim)).cuda().requires_grad_()
        self.global_encoding_optimizer_1 = torch.optim.Adam([{'params': [self.global_encoding_1]}], lr=0.001, eps=1e-15)
        self.global_encoding_2 = torch.normal(mean=0, std=0.01, size=(len(train_dataset["images"]), encoding_dim)).cuda().requires_grad_()
        self.global_encoding_optimizer_2 = torch.optim.Adam([{'params': [self.global_encoding_2]}], lr=0.001, eps=1e-15)

    @classmethod
    def get_method_info(cls):
        return MethodInfo(
            method_id="",
            required_features=frozenset(("color", "points3D_xyz")),
            supported_camera_models=frozenset(("pinhole",)),
            supported_outputs=("color",),
            viewer_default_resolution=768,
        )

    def get_info(self) -> ModelInfo:
        hparams = flatten_hparams(dict(itertools.chain(vars(self.dataset).items(), vars(self.opt).items(), vars(self.pipe).items())))
        for k in ("source_path", "resolution", "eval", "images", "model_path", "data_device"):
            hparams.pop(k, None)
        return ModelInfo(
            num_iterations=self.opt.iterations,
            loaded_step=self._loaded_step,
            loaded_checkpoint=self.checkpoint,
            hparams=hparams,
            **self.get_method_info(),
        )

    def _build_scene(self, dataset, gaussians):
        opt = copy.copy(self.dataset)
        with tempfile.TemporaryDirectory() as td:
            os.mkdir(td + "/sparse")
            opt.source_path = td  # To trigger colmap loader
            opt.model_path = td if dataset is not None else str(self.checkpoint)
            backup = sceneLoadTypeCallbacks["Colmap"]
            try:
                info = self.get_info()
                def colmap_loader(*args, **kwargs):
                    del args, kwargs
                    return _convert_dataset_to_gaussian_splatting(dataset, td, white_background=self.dataset.white_background, scale_coords=self.dataset.scale_coords)
                sceneLoadTypeCallbacks["Colmap"] = colmap_loader
                loaded_step = info.get("loaded_step")
                scene = Scene(opt, gaussians, load_iteration=str(loaded_step) if dataset is None else None)
                # NOTE: This is a hack to match the RNG state of GS on 360 scenes
                _tmp = list(range((len(next(iter(scene.train_cameras.values()))) + 6) // 7))
                random.shuffle(_tmp)
                return scene
            finally:
                sceneLoadTypeCallbacks["Colmap"] = backup

    def _format_output(self, output, options):
        del options
        return {
            k: v.cpu().numpy() for k, v in output.items()
        }

    def optimize_embedding(self, dataset, *, embedding):
        device = self.gaussians_1.get_xyz.device
        camera = dataset["cameras"].item()
        assert np.all(camera.camera_models == camera_model_to_int("pinhole")), "Only pinhole cameras supported"

        viewpoint_info = _load_caminfo(0, camera.poses, camera.intrinsics, f"{0:06d}.png", camera.image_sizes, scale_coords=self.dataset.scale_coords)
        viewpoint_cam = loadCam(self.dataset, 0, viewpoint_info, 1.0)

        if True:
            global_encoding_np_1 = self._optimize_single_gaussian(self.gaussians_1, viewpoint_cam, dataset, device)
            global_encoding_np_2 = self._optimize_single_gaussian(self.gaussians_2, viewpoint_cam, dataset, device)
            return {
                "embedding": (global_encoding_np_1, global_encoding_np_2),
                # "metrics": {
                #     "psnr": psnrs,
                #     "mse": mses,
                #     "loss": losses,
                # }
            }
        else:
            raise NotImplementedError("Trying to optimize embedding with appearance_enabled=False")

    def _optimize_single_gaussian(self, gaussians, viewpoint_cam, dataset, device):
        gaussians.freeze()
        i = 0
        losses, psnrs, mses = [], [], []

        # todo debug
        # global_encoding = (
        #     torch.from_numpy(embedding).to(device) if embedding is not None else self.gaussians.get_global_encoding
        # )
        global_encoding_param = torch.nn.Parameter(torch.zeros_like(self.global_encoding_1[0]).to(device).requires_grad_(True))
        # todo add this to config learning rate
        optimizer = torch.optim.Adam([global_encoding_param], lr=0.1)

        gt_image = torch.tensor(convert_image_dtype(dataset["images"][i], np.float32), dtype=torch.float32, device=device).permute(2, 0, 1)
        gt_mask = torch.tensor(convert_image_dtype(dataset["sampling_masks"][i], np.float32), dtype=torch.float32, device=device)[..., None].permute(2, 0, 1) if dataset["sampling_masks"] is not None else None

        with torch.enable_grad():
            # todo add this to config
            app_optim_type = 'dssim+l1'
            loss_mult = None
            if app_optim_type.endswith("-scaled"):
                app_optim_type = app_optim_type[:-7]
                # if self.model.uncertainty_model is not None:
                #     _, _, loss_mult = self.model.uncertainty_model.get_loss(gt_image, gt_image)
                #     loss_mult = (loss_mult > 1).to(dtype=loss_mult.dtype)
            # todo add this to config
            global_encoding_optim_iters = 128
            for _ in range(global_encoding_optim_iters):
                optimizer.zero_grad()

                bg = torch.zeros((3,), dtype=torch.float32, device="cuda")
                image = self._render_with_appearance_encoding(viewpoint_cam, gaussians, global_encoding_param, bg)["render"]

                if gt_mask is not None:
                    image = scale_grads(image, gt_mask.float())
                if loss_mult is not None:
                    image = scale_grads(image, loss_mult)

                mse = torch.nn.functional.mse_loss(image, gt_image)

                if app_optim_type == "mse":
                    loss = mse
                elif app_optim_type == "dssim+l1":
                    Ll1 = torch.nn.functional.l1_loss(image, gt_image)
                    ssim_value = ssim(image, gt_image, size_average=True)
                    loss = (
                            (1.0 - self.opt.lambda_dssim) * Ll1 +
                            self.opt.lambda_dssim * (1.0 - ssim_value)
                    )
                else:
                    raise ValueError(f"Unknown appearance optimization type {app_optim_type}")
                loss.backward()
                # TODO: use uncertainty here as well
                # print(float(global_encoding_param.grad.abs().max().cpu()), float(mse.cpu()))
                optimizer.step()

                losses.append(loss.detach().cpu().item())
                mses.append(mse.detach().cpu().item())
                psnrs.append(20 * math.log10(1.0) - 10 * torch.log10(mse).detach().cpu().item())

        if gaussians.optimizer is not None:
            gaussians.optimizer.zero_grad()
        global_encoding = global_encoding_param
        global_encoding_np = global_encoding_param.detach().cpu().numpy()

        torch.cuda.empty_cache()

        gaussians.unfreeze()

        return global_encoding_np

    def render(self, camera: Cameras, *, options=None) -> RenderOutput:
        camera = camera.item()
        assert np.all(camera.camera_models == camera_model_to_int("pinhole")), "Only pinhole cameras supported"

        with torch.no_grad():
            viewpoint_cam = _load_caminfo(0, camera.poses, camera.intrinsics, f"{0:06d}.png", camera.image_sizes, scale_coords=self.dataset.scale_coords)
            viewpoint = loadCam(self.dataset, 0, viewpoint_cam, 1.0)

            if options is not None:
                _np_embedding_1, _np_embedding_2 = (options or {}).get("embedding", None)
            else:
                encoding_dim = self.global_encoding_1[0].size(0)
                _np_embedding_1= np.zeros((1, encoding_dim), dtype=np.float32)
                _np_embedding_2= np.zeros((1, encoding_dim), dtype=np.float32)
            global_encoding_1 = torch.from_numpy(_np_embedding_1).cuda()
            global_encoding_2 = torch.from_numpy(_np_embedding_2).cuda()

            image_1 = torch.clamp(self._render_with_appearance_encoding(viewpoint, self.gaussians_1, global_encoding_1, self.background)["render"], 0.0, 1.0)
            image_2 = torch.clamp(self._render_with_appearance_encoding(viewpoint, self.gaussians_2, global_encoding_2, self.background)["render"], 0.0, 1.0)
            image = (image_1 + image_2) * 0.5
            color = image.detach().permute(1, 2, 0)
            return self._format_output({"color": color}, options)

    def train_iteration(self, step):
        self.step = step
        iteration = step + 1  # Gaussian Splatting is 1-indexed
        del step

        viewpoint_cam_1, global_encoding_1 = _pre_iteratrion(iteration, self.gaussians_1, self._viewpoint_stack_1, self.scene_1, self.global_encoding_1)
        viewpoint_cam_2, global_encoding_2 = _pre_iteratrion(iteration, self.gaussians_2, self._viewpoint_stack_2, self.scene_2, self.global_encoding_2)
        bg = torch.rand((3), device="cuda") if self.opt.random_background else self.background

        # Render 1
        render_pkg_1 = self._render_with_appearance_encoding(viewpoint_cam_1, self.gaussians_1, global_encoding_1, bg)
        image_1, viewspace_point_tensor_1, visibility_filter_1, radii_1 = render_pkg_1["render"], render_pkg_1["viewspace_points"], render_pkg_1["visibility_filter"], render_pkg_1["radii"]

        render_pkg_raw_1 = render(viewpoint_cam_1, self.gaussians_1, self.pipe, bg)
        image_raw_render_1 = render_pkg_raw_1["render"]

        # Loss 1
        gt_image_1, Ll1_1, ssim_value_1 = self._loss_wrapper(image_1, image_raw_render_1, viewpoint_cam_1)
        loss = (1.0 - self.opt.lambda_dssim) * Ll1_1 + self.opt.lambda_dssim * (1.0 - ssim_value_1)

        # Render 2
        render_pkg_2 = self._render_with_appearance_encoding(viewpoint_cam_2, self.gaussians_2, global_encoding_2, bg)
        image_2, viewspace_point_tensor_2, visibility_filter_2, radii_2 = render_pkg_2["render"], render_pkg_2["viewspace_points"], render_pkg_2["visibility_filter"], render_pkg_2["radii"]

        render_pkg_raw_2 = render(viewpoint_cam_2, self.gaussians_2, self.pipe, bg)
        image_raw_render_2 = render_pkg_raw_2["render"]

        # Loss 2
        gt_image_2, Ll1_2, ssim_value_2 = self._loss_wrapper(image_2, image_raw_render_2, viewpoint_cam_2)
        loss += (1.0 - self.opt.lambda_dssim) * Ll1_2 + self.opt.lambda_dssim * (1.0 - ssim_value_2)

        # cross supervision
        loss += l1_loss(render(viewpoint_cam_2, self.gaussians_2, self.pipe, bg)["render"], render(viewpoint_cam_2, self.gaussians_1, self.pipe, bg)["render"])
        loss += l1_loss(render(viewpoint_cam_1, self.gaussians_2, self.pipe, bg)["render"], render(viewpoint_cam_1, self.gaussians_1, self.pipe, bg)["render"])

        loss.backward()

        with torch.no_grad():
            psnr_value_1 = 10 * torch.log10(1 / torch.mean((image_1 - gt_image_1) ** 2))
            metrics = {
                "l1_loss_1": Ll1_1.detach().cpu().item(),
                "l1_loss_2": Ll1_2.detach().cpu().item(),
                "loss": loss.detach().cpu().item(),
                "psnr": psnr_value_1.detach().cpu().item(),
            }

            # Densification
            self._densification(iteration, self.gaussians_1, visibility_filter_1, radii_1, viewspace_point_tensor_1, self.scene_1)
            self._densification(iteration, self.gaussians_2, visibility_filter_2, radii_2, viewspace_point_tensor_2, self.scene_2)

            # Optimizer step
            if iteration < self.opt.iterations + 1:
                self.gaussians_1.optimizer.step()
                self.gaussians_1.optimizer.zero_grad(set_to_none=True)
                self.gaussians_2.optimizer.step()
                self.gaussians_2.optimizer.zero_grad(set_to_none=True)
                self.appearance_transform_optimizer.step()
                self.appearance_transform_optimizer.zero_grad(set_to_none=True)
                self.global_encoding_optimizer_1.step()
                self.global_encoding_optimizer_1.zero_grad(set_to_none=True)
                self.global_encoding_optimizer_2.step()
                self.global_encoding_optimizer_2.zero_grad(set_to_none=True)

        torch.cuda.empty_cache()

        self.step = self.step + 1
        return metrics

    def _loss_wrapper(self, image_color_tuned_render, image_raw_render, viewpoint_cam):
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        # sampling_mask = viewpoint_cam.sampling_mask.cuda() if viewpoint_cam.sampling_mask is not None else None
        sampling_mask = torch.from_numpy(self.colmap_masks[viewpoint_cam.image_name]).to(gt_image.dtype).cuda()

        # Apply mask
        if sampling_mask is not None:
            image_color_tuned_render = image_color_tuned_render * sampling_mask + (1.0 - sampling_mask) * image_color_tuned_render.detach()

        Ll1 = l1_loss(image_color_tuned_render, gt_image)
        # ssim_value = ssim(image, gt_image)
        ssim_value = ssim(image_raw_render, gt_image)
        return gt_image, Ll1, ssim_value

    def _densification(self, iteration, gaussians, visibility_filter, radii, viewspace_point_tensor, correspond_scene):
        if iteration < self.opt.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.005, correspond_scene.cameras_extent, size_threshold)

            if iteration % self.opt.opacity_reset_interval == 0 or (
                    self.dataset.white_background and iteration == self.opt.densify_from_iter):
                gaussians.reset_opacity()

    # todo save two Gaussians
    def save(self, path: str):
        self.gaussians_1.save_ply(os.path.join(str(path), f"point_cloud/iteration_{self.step}", "point_cloud.ply"))
        torch.save((self.gaussians_1.capture(), self.step), str(path) + f"/chkpnt-{self.step}.pth")
        with open(str(path) + "/args.txt", "w", encoding="utf8") as f:
            f.write(" ".join(shlex.quote(x) for x in self._args_list))

    def export_gaussian_splats(self, *, options=None):
        del options
        return dict(
            means=self.gaussians_1.get_xyz.detach().cpu().numpy(),
            scales=self.gaussians_1.get_scaling.detach().cpu().numpy(),
            opacities=self.gaussians_1.get_opacity.detach().cpu().numpy(),
            quaternions=self.gaussians_1.get_rotation.detach().cpu().numpy(),
            spherical_harmonics=self.gaussians_1.get_features.transpose(1, 2).detach().cpu().numpy())

    def _render_with_appearance_encoding(self, viewpoint_cam, gaussians, global_encoding, bg):
        # Evaluate color
        shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree + 1) ** 2)
        dir_pp = (gaussians.get_xyz - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
        eval_color = torch.clamp_min(sh2rgb + 0.5, 0.0)

        # color transform given appearance encodings
        override_color = self.appearance_transform(eval_color, global_encoding.repeat(eval_color.size(0),1), gaussians.get_local_encoding).clamp(min=0.0, max=1.0)

        # rendering with transformed color
        return render(viewpoint_cam, gaussians, self.pipe, bg, override_color=override_color)