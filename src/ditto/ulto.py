import torch as th
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from kitsu.trainer import BasePreprocessor, BaseTrainer
from kitsu.utils import instantiate_from_config


class ULTO(nn.Module):
    def __init__(self, encoder, unet, decoder) -> None:
        super().__init__()
        self.encoder = instantiate_from_config(encoder) if isinstance(encoder, dict) else encoder
        self.unet = instantiate_from_config(unet) if isinstance(unet, dict) else unet
        self.decoder = instantiate_from_config(decoder) if isinstance(decoder, dict) else decoder

    def encode(self, x):
        xyz = x[..., :3]
        xyz, c, grid = self.encoder(x)  # b n c, 3 x (b c r r), 3 x (b 1 n)
        grid, xyz, c = self.unet(grid, c, xyz)  # 3 x (b c r r)
        return grid, xyz, c

    def decode(self, grid, query, xyz, c):
        out = self.decoder(grid, query, xyz, c)  # b m dim_out
        return out

    def forward(self, query, x):
        """
        - query: b m 3
        - x: b n dim_in
        """
        grid, xyz, c = self.encode(x)
        out = self.decode(grid, query, xyz, c)
        return out


class Preprocessor(BasePreprocessor):
    def __init__(self, device) -> None:
        super().__init__(device)

    def __call__(self, batch, augmentation=False):
        s = EasyDict(log={})
        """
        points                      : torch.Size([1, 2048, 3])*
        points.occ                  : torch.Size([1, 2048])*
        points.sub_points_idx       : torch.Size([1, 2048])
        inputs                      : torch.Size([1, 3000, 3])*
        inputs.normals              : torch.Size([1, 3000, 3])
        
        # for evaluation
        idx                         : torch.Size([1])
        points_iou                  : torch.Size([1, 100000, 3])
        points_iou.occ              : torch.Size([1, 100000])
        pointcloud_chamfer          : torch.Size([1, 100000, 3])
        pointcloud_chamfer.normals  : torch.Size([1, 100000, 3])
        """
        for k, v in batch.items():
            s[k.replace(".", "_")] = v.to(self.device, non_blocking=True)
        s.n = s.points.size(0)
        return s


class Trainer(BaseTrainer):
    def __init__(
        self,
        # boilerplate arguments
        args,
        n_samples_per_class: int = 10,
        find_unused_parameters: bool = False,
        sample_at_least_per_epochs: int = None,
        mixed_precision: bool = False,
        clip_grad: float = 0,
        num_saves: int = 5,
        epochs_to_save: int = 0,
        use_sync_bn: bool = False,
        monitor: str = "loss",
        small_is_better: bool = True,
        use_sam: bool = False,
        use_esam: bool = False,
        save_only_improved: bool = True,
        tqdm_ncols: int = 128,
        gradient_accumulation_steps=1,
        # custom arguments
        loss_type="sum",
        ckpt=None,
        start_epoch=None,
    ):
        self.loss_type = loss_type
        self.ckpt = ckpt
        self.start_epoch = start_epoch

        super().__init__(
            args,
            n_samples_per_class=n_samples_per_class,
            find_unused_parameters=find_unused_parameters,
            sample_at_least_per_epochs=sample_at_least_per_epochs,
            mixed_precision=mixed_precision,
            clip_grad=clip_grad,
            num_saves=num_saves,
            epochs_to_save=epochs_to_save,
            use_sync_bn=use_sync_bn,
            monitor=monitor,
            small_is_better=small_is_better,
            use_sam=use_sam,
            use_esam=use_esam,
            save_only_improved=save_only_improved,
            tqdm_ncols=tqdm_ncols,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

    def build_dataset(self):
        self.dl_train, self.dl_valid, self.dl_test = instantiate_from_config(self.args.dataset, cfg=self.args.cfg)

    def build_network(self):
        super().build_network()

        # load checkpoint
        if self.ckpt is not None:
            self.log.info("Load checkpoint:", self.ckpt)
            ckpt = th.load(self.ckpt, map_location="cpu")
            self.model_src.load_state_dict(ckpt["model"])
            self.optim.load_state_dict(ckpt["optim"])
            self.log.info("Start epoch from:", ckpt["epoch"])
            self.epoch = int(ckpt["epoch"])

        if self.start_epoch is not None:
            self.log.info("Start epoch from:", self.start_epoch)
            self.epoch = int(self.start_epoch)

    def step(self, s):
        occ = self.model_optim(s.points, s.inputs).flatten(1)  # b m
        if self.loss_type == "sum":
            s.log.loss = F.binary_cross_entropy_with_logits(occ, s.points_occ, reduction="none").sum(-1).mean()
        elif self.loss_type == "mean":
            s.log.loss = F.binary_cross_entropy_with_logits(occ, s.points_occ)
        else:
            raise NotImplementedError(self.loss_type)

        with th.no_grad():
            pred_t = occ.sigmoid() > 0.5
            real_t = s.points_occ > 0.5

            tp = pred_t & real_t
            union = pred_t | real_t

            tp_sum = tp.float().sum(1)
            union_sum = union.float().sum(1)
            real_t_sum = real_t.float().sum(1)
            pred_t_sum = pred_t.float().sum(1)

            iou = (tp_sum / union_sum).nan_to_num_(0, 0, 0)
            p = (tp_sum / pred_t_sum).nan_to_num_(0, 0, 0)
            r = (tp_sum / real_t_sum).nan_to_num_(0, 0, 0)
            f1 = (2 * p * r / (p + r)).nan_to_num_(0, 0, 0)

            s.log.iou = iou.mean()
            s.log.p = p.mean()
            s.log.r = r.mean()
            s.log.f1 = f1.mean()
