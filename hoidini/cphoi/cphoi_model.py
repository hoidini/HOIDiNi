from torch import nn
import torch
from torch import Tensor
from hoidini.amasstools.geometry import matrix_to_euler_angles, matrix_to_rotation_6d
from hoidini.closd.diffusion_planner.model.BERT.BERT_encoder import load_bert
from hoidini.closd.diffusion_planner.model.mdm import (
    InputProcess,
    OutputProcess,
    PositionalEncoding,
    TimestepEmbedder,
)
from hoidini.object_conditioning.object_encoder_pointwise import (
    ObjectPointwiseEncoder,
    pyg_batch_to_torch_batch,
)


class CPHOI(nn.Module):
    def __init__(self, pred_len, context_len, n_feats, num_layers, cond_mask_prob):
        super().__init__()

        self.latent_dim = 512
        self.num_heads = 4
        self.num_layers = num_layers
        self.ff_size = 1024
        self.dropout = 0.1
        self.activation = "gelu"
        self.cond_mode = "text"

        self.pred_len = pred_len
        self.context_len = context_len
        self.total_len = self.pred_len + self.context_len
        self.cond_mask_prob = cond_mask_prob
        self.input_feats = n_feats

        self.input_process = InputProcess(
            data_rep="hml_vec", input_feats=self.input_feats, latent_dim=self.latent_dim
        )
        self.output_process = OutputProcess(
            "hml_vec", self.input_feats, self.latent_dim, self.input_feats, 1
        )

        seqTransDecoderLayer = nn.TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
        )
        self.seqTransDecoder = nn.TransformerDecoder(
            seqTransDecoderLayer, num_layers=self.num_layers
        )
        self.sequence_pos_encoder = PositionalEncoding(
            self.latent_dim, self.dropout, max_len=5000
        )

        self.embed_timestep = TimestepEmbedder(
            self.latent_dim, self.sequence_pos_encoder
        )

        self.text_encoder = load_bert("distilbert/distilbert-base-uncased")
        self.embed_text = nn.Linear(768, self.latent_dim)

        self.object_encoder = ObjectPointwiseEncoder(self.latent_dim)
        self.spatial_encoding = SpatialEncoding()
        self.embed_obj = nn.Linear(
            self.latent_dim + self.spatial_encoding.output_dim(), self.latent_dim
        )

        self.init_tfm_encoder = TransformEmbedding(self.latent_dim)

        # self.table_encoder = nn.Linear(1, self.latent_dim)

        self.time_type_embedding = nn.Parameter(0.1 * torch.randn(self.latent_dim))
        self.text_type_embedding = nn.Parameter(0.1 * torch.randn(self.latent_dim))
        self.points_type_embedding = nn.Parameter(0.1 * torch.randn(self.latent_dim))
        self.init_tfm_type_embedding = nn.Parameter(0.1 * torch.randn(self.latent_dim))

    def freeze_object_encoder(self):
        for param in self.object_encoder.parameters():
            param.requires_grad = False

    def parameters_wo_clip(self):
        return [
            p
            for name, p in self.named_parameters()
            if not name.startswith("clip_model.")
        ]

    def encode_text(self, raw_text):
        enc_text, mask = self.text_encoder(
            raw_text
        )  # self.clip_model.get_last_hidden_state(raw_text, return_mask=True)  # mask: False means no token there
        enc_text = enc_text.permute(1, 0, 2)
        mask = (
            ~mask
        )  # mask: True means no token there, we invert since the meaning of mask for transformer is inverted  https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        return enc_text, mask

    def forward(self, x, timesteps, y) -> Tensor:
        """
        x: (bs, n_feats, 1, seq_len)
        comment: in the code base the naming is (bs, njoints, nfeats, nframes)
        """
        bs = x.shape[0]

        x = torch.cat([y["prefix"], x], dim=-1)
        x = self.input_process(x)
        seq_len = x.shape[0]
        x = self.sequence_pos_encoder(x)

        # Mask for prefix completion
        y["mask"] = torch.cat(
            [
                torch.ones(
                    [bs, 1, 1, self.context_len],
                    dtype=y["mask"].dtype,
                    device=y["mask"].device,
                ),
                y["mask"],
            ],
            dim=-1,
        )

        ##################################
        # Get time embeddings
        ##################################
        time_emb = self.embed_timestep(timesteps)  # (1, bs, d_time)

        ##################################
        # Get text embeddings
        ##################################
        if "text_embed" in y.keys():  # caching option
            enc_text, mask_text = y["text_embed"]
        else:
            enc_text, mask_text = self.encode_text(y["text"])  # (seq_len, bs, d_text)

        force_mask = y.get("text_uncond", False)  # for CFG
        enc_text = self.mask_cond(
            enc_text, force_mask=force_mask
        )  # mask the entire text for CFG (null conditioning) or element-wise with cond_mask_prob
        text_emb = self.embed_text(enc_text)  # (n_tokens, bs, latent_dim)

        ##################################
        # Get object embeddings
        ##################################
        pyg_input_data = y["obj_points"]
        if "obj_emb" in y.keys():  # caching option
            obj_emb = y["obj_emb"]  # (bs, n_points, d)
        else:
            obj_emb = self.object_encoder(pyg_input_data)  # (bs, n_points, d_pointnet)
        points_xyz = pyg_batch_to_torch_batch(
            pyg_input_data, "pos"
        )  # (bs, n_points, 3)
        spatial_emb = self.spatial_encoding(points_xyz)  # (bs, n_points, d_spatial)
        obj_emb = torch.cat([obj_emb, spatial_emb], dim=-1)  # (bs, n_points, d_obj)
        if "is_zero_hoi_mask" in y:
            obj_emb = obj_emb * y["is_zero_hoi_mask"].reshape(
                bs, 1, 1
            )  # zero out the point embeddings of non GRAB datapoints

        obj_emb = obj_emb.permute(1, 0, 2)  # (n_points, bs, d_obj)
        obj_emb = self.embed_obj(obj_emb)  # (n_points, bs, latent_dim)

        ##################################
        # Get initial transform embeddings
        ##################################
        init_tfm_emb = self.init_tfm_encoder(y["tfms_root_global"]).unsqueeze(
            0
        )  # (1, bs, latent_dim
        assert init_tfm_emb.shape == (1, bs, self.latent_dim)

        ##################################
        # Combine all embeddings
        ##################################
        time_emb = time_emb + self.time_type_embedding  # (1, bs, latent_dim)
        obj_emb = obj_emb + self.points_type_embedding  # (n_points, bs, latent_dim)
        text_emb = text_emb + self.text_type_embedding  # (n_tokens, bs, latent_dim)
        init_tfm_emb = (
            init_tfm_emb + self.init_tfm_type_embedding
        )  # (1, bs, latent_dim)

        cond_emb = torch.cat(
            [time_emb, init_tfm_emb, obj_emb, text_emb], dim=0
        )  # (1 + n_tokens + n_points, bs, latent_dim)
        # cond_emb = self.cond_norm(cond_emb)
        mask_time = torch.zeros((bs, 1), dtype=torch.bool, device=cond_emb.device)
        mask_init_tfm = torch.zeros((bs, 1), dtype=torch.bool, device=cond_emb.device)
        mask_obj = torch.zeros(
            (bs, obj_emb.shape[0]), dtype=torch.bool, device=cond_emb.device
        )
        cond_mask = torch.cat(
            [mask_time, mask_init_tfm, mask_obj, mask_text], dim=1
        )  # (bs, 1 + n_points + n_tokens, bs, 1)

        frames_mask = None
        is_valid_mask = (
            y["mask"].shape[-1] > 1
        )  # Don't use mask with the generate script
        if is_valid_mask:
            frames_mask = y["mask"][..., :seq_len]  # not :x.shape[0] or :bs
            frames_mask = frames_mask.squeeze(1).squeeze(1)
            frames_mask = torch.logical_not(frames_mask).to(x.device)

        output = self.seqTransDecoder(
            tgt=x,
            memory=cond_emb,
            memory_key_padding_mask=cond_mask,
            tgt_key_padding_mask=frames_mask,
        )

        output = output[self.context_len :]
        y["mask"] = y["mask"][..., self.context_len :]

        output = self.output_process(output)
        return output

    def mask_cond(self, cond, force_mask=False):
        seq_len, bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            ).view(
                1, bs, 1
            )  # 1-> use null_cond, 0-> use real cond
            return cond * (1.0 - mask)
        else:
            return cond


class SpatialEncoding(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        num_freqs: int = 10,
        include_input: bool = True,
        log_scale: bool = True,
    ):
        """
        Args:
            input_dim: Dimensionality of the input (e.g., 3 for 3D positions)
            num_freqs: Number of frequency bands (L in NeRF)
            include_input: Whether to include the raw input in the output
            log_scale: Whether to use frequencies spaced in log scale (as in NeRF)
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_freqs = num_freqs
        self.include_input = include_input
        if log_scale:
            self.freq_bands = 2.0 ** torch.arange(num_freqs)
        else:
            self.freq_bands = torch.linspace(1.0, 2.0 ** (num_freqs - 1), num_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (..., input_dim)
        Returns:
            Tensor of shape (..., input_dim * (2 * num_freqs + [1 if include_input]))
        """
        out = [x] if self.include_input else []
        for freq in self.freq_bands.to(x.device):
            for fn in [torch.sin, torch.cos]:
                out.append(fn(x * freq * torch.pi))
        return torch.cat(out, dim=-1)

    def output_dim(self):
        return self.input_dim * (2 * self.num_freqs + int(self.include_input))


class TransformEmbedding(nn.Module):
    """
    Embed 4x4 transformation matrix into a latent space.
    """

    def __init__(self, latent_dim, num_freqs=10):
        super().__init__()
        self.spatial_encoding = SpatialEncoding(num_freqs=num_freqs, include_input=True)
        cont6d_dim = 6
        euler_dim = 3
        tfm_feat_dim = self.spatial_encoding.output_dim() + cont6d_dim + euler_dim
        self.linear = nn.Linear(tfm_feat_dim, latent_dim)

    def forward(self, tfm):
        # tfm: (B,4,4)
        assert tfm.shape[1:] == (4, 4), f"Expected (B,4,4), got {tfm.shape}"
        bs = tfm.shape[0]
        rot = tfm[:, :3, :3]
        trans = tfm[:, :3, 3]

        emb_list = [
            self.spatial_encoding(trans),
            matrix_to_rotation_6d(rot),
            matrix_to_euler_angles(rot, "XYZ"),
        ]
        init_emb = torch.cat(emb_list, dim=-1)  # (B, tfm_feat_dim)
        init_emb = self.linear(init_emb)  # (B, latent_dim)

        assert init_emb.shape == (bs, self.linear.out_features)
        return init_emb
