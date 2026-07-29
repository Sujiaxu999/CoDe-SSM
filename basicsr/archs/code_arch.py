import math
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from basicsr.archs.arch_util import trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY



def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)

        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        # Normalize in float32 for numerical stability.
        x_fp32 = x.float()
        norm = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + 1e-6)
        return norm.to(x.dtype) * self.weight + self.bias


class ImageLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = BiasLayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# =========================================================
# Prototype clustering front-end
# =========================================================
class PrototypeClusterAssignment(nn.Module):

    def __init__(self, dim, num_clusters, tau_init=1.0, tau_min=0.1):
        super().__init__()
        self.num_clusters = num_clusters
        self.tau_min = tau_min

        self.prototype = nn.Parameter(torch.randn(num_clusters, dim) * 0.02)
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))

    @property
    def tau(self):
        return torch.exp(self.log_tau).clamp(min=self.tau_min)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).transpose(1, 2)

        x_norm = F.normalize(x_flat, dim=-1, eps=1e-8)
        prototype_norm = F.normalize(self.prototype, dim=-1, eps=1e-8)

        similarity = torch.einsum("bnc,kc->bnk", x_norm, prototype_norm) / self.tau
        similarity = similarity.clamp(-30.0, 30.0)

        position_assignment = F.softmax(similarity.float(), dim=-1).to(x_norm.dtype)
        position_assignment = position_assignment.transpose(1, 2)

        # NaN guard for low-light / zero-signal regions.
        position_assignment = torch.nan_to_num(
            position_assignment,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        assignment_norm = position_assignment / (
            position_assignment.sum(dim=-1, keepdim=True).clamp(min=1e-4) + 1e-6
        )

        cluster_centers = torch.bmm(assignment_norm, x_flat.float()).to(x_flat.dtype)

        return cluster_centers, position_assignment

    @staticmethod
    def reconstruct(cluster_centers, position_assignment, shape):
        B, C, H, W = shape
        out = torch.bmm(position_assignment.transpose(1, 2), cluster_centers)
        return out.transpose(1, 2).view(B, C, H, W)


class PrototypeClusteringFrontend(nn.Module):

    def __init__(self, dim, num_clusters):
        super().__init__()
        self.cluster_assignment = PrototypeClusterAssignment(dim, num_clusters)

    def forward(self, x):
        cluster_centers, assignment = self.cluster_assignment(x)

        reconstruction = PrototypeClusterAssignment.reconstruct(
            cluster_centers,
            assignment,
            x.shape,
        )

        detail_residual = x - reconstruction
        detail_residual = torch.nan_to_num(
            detail_residual,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return assignment, detail_residual


# =========================================================
# Local High-Frequency Module, LHFM
# Detail recovery pathway
# =========================================================
class LocalHighFrequencyModule(nn.Module):


    def __init__(
        self,
        dim,
        num_experts=4,
        top_k=2,
        expert_ratio=0.5,
        router_hidden_ratio=0.125,
    ):
        super().__init__()

        self.num_experts = max(1, num_experts)
        self.top_k = max(1, min(top_k, self.num_experts))

        expert_dim = max(int(dim * expert_ratio), 16)
        router_hidden = max(int(dim * router_hidden_ratio), 8)

        laplacian_kernel = torch.tensor(
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        sobel_x_kernel = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        sobel_y_kernel = sobel_x_kernel.transpose(-1, -2).contiguous()

        self.register_buffer(
            "hfe_laplacian_weight",
            laplacian_kernel.repeat(dim, 1, 1, 1),
        )
        self.register_buffer(
            "hfe_sobel_x_weight",
            sobel_x_kernel.repeat(dim, 1, 1, 1),
        )
        self.register_buffer(
            "hfe_sobel_y_weight",
            sobel_y_kernel.repeat(dim, 1, 1, 1),
        )

        self.hfe_scale = nn.Parameter(torch.tensor(1.0))
        self.output_scale = nn.Parameter(torch.tensor(0.1))

        self.channel_reduction = nn.Conv2d(dim, expert_dim, 1, bias=False)

        base_experts = [
            nn.Sequential(
                nn.Conv2d(
                    expert_dim,
                    expert_dim,
                    3,
                    1,
                    1,
                    groups=expert_dim,
                    bias=False,
                ),
                nn.SiLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(
                    expert_dim,
                    expert_dim,
                    3,
                    1,
                    2,
                    dilation=2,
                    groups=expert_dim,
                    bias=False,
                ),
                nn.SiLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(
                    expert_dim,
                    expert_dim,
                    (1, 5),
                    1,
                    (0, 2),
                    groups=expert_dim,
                    bias=False,
                ),
                nn.Conv2d(
                    expert_dim,
                    expert_dim,
                    (5, 1),
                    1,
                    (2, 0),
                    groups=expert_dim,
                    bias=False,
                ),
                nn.SiLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(
                    expert_dim,
                    expert_dim,
                    5,
                    1,
                    2,
                    groups=expert_dim,
                    bias=False,
                ),
                nn.SiLU(inplace=True),
            ),
        ]

        if self.num_experts > len(base_experts):
            for _ in range(self.num_experts - len(base_experts)):
                base_experts.append(
                    nn.Sequential(
                        nn.Conv2d(
                            expert_dim,
                            expert_dim,
                            3,
                            1,
                            1,
                            groups=expert_dim,
                            bias=False,
                        ),
                        nn.SiLU(inplace=True),
                    )
                )
        elif self.num_experts < len(base_experts):
            base_experts = base_experts[: self.num_experts]

        self.experts = nn.ModuleList(base_experts)

        self.channel_expansion = nn.Conv2d(expert_dim, dim, 1, bias=False)

        self.expert_router = nn.Sequential(
            nn.Conv2d(dim + 1, router_hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(router_hidden, self.num_experts, 1, bias=True),
        )

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, max(dim // 4, 8), 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(dim // 4, 8), dim, 1, bias=False),
            nn.Sigmoid(),
        )

        self.last_lhfm_balance_loss = None

    def high_frequency_energy_filter(self, reference_feature):

        x_fp32 = reference_feature.float()

        laplacian_response = torch.abs(
            F.conv2d(
                x_fp32,
                self.hfe_laplacian_weight.float(),
                padding=1,
                groups=reference_feature.shape[1],
            )
        )

        gradient_x = F.conv2d(
            x_fp32,
            self.hfe_sobel_x_weight.float(),
            padding=1,
            groups=reference_feature.shape[1],
        )

        gradient_y = F.conv2d(
            x_fp32,
            self.hfe_sobel_y_weight.float(),
            padding=1,
            groups=reference_feature.shape[1],
        )

        gradient_magnitude = torch.sqrt(gradient_x.pow(2) + gradient_y.pow(2) + 1e-8)

        energy = (laplacian_response + gradient_magnitude).mean(dim=1, keepdim=True)
        energy_mean = energy.mean(dim=(2, 3), keepdim=True).clamp(min=1e-8)
        energy_norm = (energy / energy_mean).clamp(max=10.0)

        return energy_norm.to(reference_feature.dtype)

    def forward(self, detail_residual, reference_feature):

        hfe = self.high_frequency_energy_filter(reference_feature)
        hfe = torch.nan_to_num(hfe, nan=0.0, posinf=0.0, neginf=0.0)

        hfe_mask = torch.sigmoid(self.hfe_scale * (hfe - 1.0))

        masked_detail = detail_residual * hfe_mask
        masked_detail = torch.nan_to_num(
            masked_detail,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        router_input = torch.cat(
            [masked_detail, hfe.to(detail_residual.dtype)],
            dim=1,
        )

        expert_logits = self.expert_router(router_input)
        expert_logits = torch.nan_to_num(
            expert_logits,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        topk_values, topk_indices = torch.topk(
            expert_logits,
            k=self.top_k,
            dim=1,
        )

        sparse_logits = torch.full_like(expert_logits, -1e3)
        sparse_logits.scatter_(1, topk_indices, topk_values)

        expert_route = F.softmax(sparse_logits.float(), dim=1).to(expert_logits.dtype)
        expert_route = torch.nan_to_num(
            expert_route,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        expert_input = self.channel_reduction(masked_detail)

        expert_output = 0
        for expert_index, expert in enumerate(self.experts):
            expert_output = expert_output + expert(expert_input) * expert_route[
                :, expert_index : expert_index + 1
            ]

        if self.training:
            with torch.no_grad():
                hard_load = (
                    torch.zeros_like(expert_route)
                    .scatter_(1, topk_indices, 1.0)
                    .mean(dim=(0, 2, 3))
                    .detach()
                )

            importance = expert_route.float().mean(dim=(0, 2, 3))
            balance_loss = self.num_experts * torch.sum(importance * hard_load)

            self.last_lhfm_balance_loss = (
                balance_loss if torch.isfinite(balance_loss) else None
            )
        else:
            self.last_lhfm_balance_loss = None

        local_output = self.channel_expansion(expert_output)
        local_output = local_output * self.channel_gate(reference_feature)

        return self.output_scale * local_output


class DetailRecoveryPathway(nn.Module):

    def __init__(
        self,
        d_inner,
        num_experts=4,
        top_k=2,
        expert_ratio=0.5,
        router_hidden_ratio=0.125,
    ):
        super().__init__()
        self.lhfm = LocalHighFrequencyModule(
            dim=d_inner,
            num_experts=num_experts,
            top_k=top_k,
            expert_ratio=expert_ratio,
            router_hidden_ratio=router_hidden_ratio,
        )

    def forward(self, detail_residual, reference_feature):
        local_output = self.lhfm(detail_residual, reference_feature)
        local_output = torch.nan_to_num(
            local_output,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return local_output


# =========================================================
# Selective SSM scan over compact cluster sequence
# =========================================================
class SelectiveClusterScan(nn.Module):

    def __init__(
        self,
        d_inner,
        d_state=8,
        delta_time_rank="auto",
        delta_time_min=0.001,
        delta_time_max=0.1,
        delta_time_init="random",
        delta_time_scale=1.0,
        delta_time_init_floor=1e-4,
    ):
        super().__init__()

        self.d_inner = d_inner
        self.d_state = d_state

        if delta_time_rank == "auto":
            self.delta_time_rank = math.ceil(d_inner / 16)
        else:
            self.delta_time_rank = int(delta_time_rank)

        self.state_input_projection_weight = nn.Parameter(
            torch.zeros(self.delta_time_rank + self.d_state * 2, d_inner)
        )
        trunc_normal_(self.state_input_projection_weight, std=0.02)

        self.delta_time_projection_weight = nn.Parameter(
            torch.zeros(d_inner, self.delta_time_rank)
        )
        self.delta_time_bias = nn.Parameter(torch.zeros(d_inner))

        self._initialize_delta_time(
            delta_time_scale,
            delta_time_init,
            delta_time_min,
            delta_time_max,
            delta_time_init_floor,
        )

        A = (
            torch.arange(1, d_state + 1, dtype=torch.float32)
            .unsqueeze(0)
            .expand(d_inner, -1)
            .contiguous()
        )
        self.a_logs = nn.Parameter(torch.log(A))
        self.a_logs._no_weight_decay = True

        self.ds = nn.Parameter(torch.ones(d_inner))
        self.ds._no_weight_decay = True

    def _initialize_delta_time(
        self,
        delta_time_scale,
        delta_time_init,
        delta_time_min,
        delta_time_max,
        delta_time_init_floor,
    ):
        delta_time_init_std = self.delta_time_rank ** -0.5 * delta_time_scale

        if delta_time_init == "random":
            nn.init.uniform_(
                self.delta_time_projection_weight,
                -delta_time_init_std,
                delta_time_init_std,
            )
        else:
            nn.init.constant_(
                self.delta_time_projection_weight,
                delta_time_init_std,
            )

        dt = torch.exp(
            torch.rand(self.d_inner)
            * (math.log(delta_time_max) - math.log(delta_time_min))
            + math.log(delta_time_min)
        ).clamp(min=delta_time_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))

        with torch.no_grad():
            self.delta_time_bias.copy_(inv_dt)

        self.delta_time_bias._no_reinit = True

    def forward(self, tokens):
        """
        tokens: B, L, C
            L = K cluster centers.
        """
        B, L, C = tokens.shape

        x = tokens.transpose(1, 2).contiguous()

        projected = torch.einsum(
            "bcl,dc->bdl",
            x,
            self.state_input_projection_weight,
        )

        delta_time, state_b, state_c = torch.split(
            projected,
            [self.delta_time_rank, self.d_state, self.d_state],
            dim=1,
        )

        delta_time = torch.einsum(
            "brl,dr->bdl",
            delta_time,
            self.delta_time_projection_weight,
        )

        x = x.float()
        delta_time = delta_time.contiguous().float()

        # Keep the group dimension expected by selective_scan_fn.
        state_b = state_b.unsqueeze(1).float().contiguous()
        state_c = state_c.unsqueeze(1).float().contiguous()

        scan_output = selective_scan_fn(
            x,
            delta_time,
            -torch.exp(self.a_logs.float()),
            state_b,
            state_c,
            self.ds.float(),
            z=None,
            delta_bias=self.delta_time_bias.float(),
            delta_softplus=True,
            return_last_state=False,
        )

        scan_output = torch.nan_to_num(
            scan_output,
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )

        return scan_output.transpose(1, 2)


# =========================================================
# Context modeling pathway
# =========================================================
class ContextModelingPathway(nn.Module):


    def __init__(
        self,
        d_inner,
        heads,
        d_state=8,
        delta_time_rank="auto",
        delta_time_min=0.001,
        delta_time_max=0.1,
        delta_time_init="random",
        delta_time_scale=1.0,
        delta_time_init_floor=1e-4,
    ):
        super().__init__()

        self.d_inner = d_inner
        self.heads = heads

        self.value_projection = nn.Conv2d(d_inner, d_inner * heads, 1)
        self.output_channel_projection = nn.Conv2d(d_inner * heads, d_inner, 1)

        self.ssm = SelectiveClusterScan(
            d_inner=d_inner,
            d_state=d_state,
            delta_time_rank=delta_time_rank,
            delta_time_min=delta_time_min,
            delta_time_max=delta_time_max,
            delta_time_init=delta_time_init,
            delta_time_scale=delta_time_scale,
            delta_time_init_floor=delta_time_init_floor,
        )

    def _aggregate_cluster_values(self, x, assignment):

        B, C, H, W = x.shape

        value = self.value_projection(x)
        value = rearrange(value, "b (e c) h w -> (b e) c h w", e=self.heads)
        value_flat = value.view(B * self.heads, self.d_inner, -1).transpose(1, 2)

        num_clusters = assignment.shape[1]

        assignment_by_head = assignment.unsqueeze(1).expand(
            -1, self.heads, -1, -1
        )
        assignment_by_head = assignment_by_head.reshape(
            B * self.heads,
            num_clusters,
            H * W,
        )

        assignment_by_head = torch.nan_to_num(
            assignment_by_head,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        normalized_assignment = assignment_by_head / (
            assignment_by_head.sum(dim=-1, keepdim=True).clamp(min=1e-6) + 1e-8
        )

        cluster_values = torch.bmm(
            normalized_assignment.float(),
            value_flat.float(),
        ).to(value_flat.dtype)

        cluster_values = torch.nan_to_num(
            cluster_values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return cluster_values, assignment_by_head

    def forward(self, x, assignment):
        """
        x: B, C, H, W
        assignment: B, K, N
        """
        B, C, H, W = x.shape

        cluster_values, assignment_by_head = self._aggregate_cluster_values(
            x,
            assignment,
        )

        scanned_cluster_values = self.ssm(cluster_values)

        global_output = torch.bmm(
            assignment_by_head.transpose(1, 2).to(scanned_cluster_values.dtype),
            scanned_cluster_values,
        )

        global_output = global_output.transpose(1, 2).view(
            B * self.heads,
            self.d_inner,
            H,
            W,
        )

        global_output = rearrange(
            global_output,
            "(b e) c h w -> b (e c) h w",
            e=self.heads,
        )

        global_output = global_output.to(x.dtype)
        global_output = self.output_channel_projection(global_output)

        global_output = torch.nan_to_num(
            global_output,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return global_output


# =========================================================
# Dual-pathway fusion
# =========================================================
class DualPathwayFusion(nn.Module):


    def __init__(self, d_inner):
        super().__init__()
        self.refinement_conv = nn.Conv2d(
            d_inner,
            d_inner,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=d_inner,
            bias=True,
        )

    def forward(self, context_feature, detail_feature):
        fused = context_feature + detail_feature
        fused = fused + self.refinement_conv(fused)
        fused = torch.nan_to_num(
            fused,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return fused


# =========================================================
# CoDe dual-pathway module
# =========================================================
class CoDeDualPathwayModule(nn.Module):

    def __init__(
        self,
        d_model,
        num_clusters=16,
        heads=1,
        d_state=8,
        d_conv=3,
        expand=2,
        delta_time_rank="auto",
        delta_time_min=0.001,
        delta_time_max=0.1,
        delta_time_init="random",
        delta_time_scale=1.0,
        delta_time_init_floor=1e-4,
        bias=False,
        conv_bias=True,
        hf_num_experts=4,
        hf_top_k=2,
        hf_expert_ratio=0.5,
        hf_router_hidden_ratio=0.125,
    ):
        super().__init__()

        self.d_inner = int(expand * d_model) // heads


        if delta_time_rank == "auto":
            ssm_delta_time_rank = math.ceil(d_model / 16)
        else:
            ssm_delta_time_rank = delta_time_rank

        # Z -> [F, G]
        self.feature_gate_projection = nn.Linear(
            d_model,
            self.d_inner * 2,
            bias=bias,
        )

        # Optional local mixer for the feature branch.
        self.feature_local_mixer = nn.Conv2d(
            self.d_inner,
            self.d_inner,
            groups=self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            bias=conv_bias,
        )

        self.activation = nn.SiLU()

        # Clustering front-end:
        # F -> assignment A + residual R
        self.clustering_frontend = PrototypeClusteringFrontend(
            dim=self.d_inner,
            num_clusters=num_clusters,
        )

        # Context pathway / GCSM:
        # F, A -> F_global
        self.context_pathway = ContextModelingPathway(
            d_inner=self.d_inner,
            heads=heads,
            d_state=d_state,
            delta_time_rank=ssm_delta_time_rank,
            delta_time_min=delta_time_min,
            delta_time_max=delta_time_max,
            delta_time_init=delta_time_init,
            delta_time_scale=delta_time_scale,
            delta_time_init_floor=delta_time_init_floor,
        )

        # Detail pathway / LHFM:
        # R, F -> F_local
        self.detail_pathway = DetailRecoveryPathway(
            d_inner=self.d_inner,
            num_experts=hf_num_experts,
            top_k=hf_top_k,
            expert_ratio=hf_expert_ratio,
            router_hidden_ratio=hf_router_hidden_ratio,
        )

        # Fuse F_global and F_local.
        self.fusion = DualPathwayFusion(self.d_inner)

        # Output gate:
        # Norm(F_out) * SiLU(G)
        self.output_norm = nn.LayerNorm(self.d_inner)
        self.output_projection = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        x_in = rearrange(x, "b c h w -> b h w c")

        # -------------------------------------------------
        # Feature/Gate projection.
        # -------------------------------------------------
        xz = self.feature_gate_projection(x_in)
        feature_branch, gate_branch = xz.chunk(2, dim=-1)

        # Local feature mixing.
        feature_branch = self.activation(
            self.feature_local_mixer(
                feature_branch.permute(0, 3, 1, 2).contiguous()
            )
        )

        # -------------------------------------------------
        # Prototype clustering front-end.
        #   A = soft assignment
        #   R = clustering residual
        # -------------------------------------------------
        assignment, detail_residual = self.clustering_frontend(feature_branch)

        # -------------------------------------------------
        # Context pathway / GCSM.
        #   F_global = Broadcast(SSM(Aggregate(F, A)), A)
        # -------------------------------------------------
        context_feature = self.context_pathway(
            feature_branch,
            assignment,
        )

        # -------------------------------------------------
        # Detail pathway / LHFM.
        #   F_local = LHFM(R, F)
        # -------------------------------------------------
        detail_feature = self.detail_pathway(
            detail_residual,
            feature_branch,
        )

        # -------------------------------------------------
        # Dual-pathway fusion.
        #   F_out = F_global + F_local
        # -------------------------------------------------
        fused_feature = self.fusion(
            context_feature,
            detail_feature,
        )

        # -------------------------------------------------
        # Gated output.
        #   F'_out = LN(F_out) * SiLU(G)
        # -------------------------------------------------
        fused_feature = fused_feature.permute(0, 2, 3, 1).contiguous()
        fused_feature = self.output_norm(fused_feature) * F.silu(gate_branch)

        out = self.output_projection(fused_feature)
        out = rearrange(out, "b h w c -> b c h w")

        return out, assignment


# =========================================================
# Gated feed-forward network, GDFN
# =========================================================
class GatedFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias=False):
        super().__init__()

        hidden_dim = int(dim * ffn_expansion_factor)

        self.input_projection = nn.Conv2d(dim, hidden_dim * 2, 1, bias=bias)
        self.depthwise_conv = nn.Conv2d(
            hidden_dim * 2,
            hidden_dim * 2,
            3,
            1,
            1,
            groups=hidden_dim * 2,
            bias=bias,
        )
        self.output_projection = nn.Conv2d(hidden_dim, dim, 1, bias=bias)

    def forward(self, x):
        x = self.input_projection(x)
        gate_input, gate_value = self.depthwise_conv(x).chunk(2, dim=1)
        return self.output_projection(F.gelu(gate_input) * gate_value)


# =========================================================
# Paper-aligned CoDeBlock
# =========================================================
class CoDeBlock(nn.Module):

    def __init__(
        self,
        use_dual_pathway,
        feature_dim,
        num_clusters,
        heads,
        ffn_expansion_factor,
        use_checkpoint=False,
        hf_num_experts=4,
        hf_top_k=2,
        hf_expert_ratio=0.5,
        hf_router_hidden_ratio=0.125,
    ):
        super().__init__()

        self.use_dual_pathway = use_dual_pathway
        self.use_checkpoint = use_checkpoint

        self.ffn_norm = ImageLayerNorm(feature_dim)
        self.ffn = GatedFeedForward(feature_dim, ffn_expansion_factor)

        if self.use_dual_pathway:
            self.dual_norm = ImageLayerNorm(feature_dim)
            self.dual_pathway = CoDeDualPathwayModule(
                d_model=feature_dim,
                num_clusters=num_clusters,
                heads=heads,
                hf_num_experts=hf_num_experts,
                hf_top_k=hf_top_k,
                hf_expert_ratio=hf_expert_ratio,
                hf_router_hidden_ratio=hf_router_hidden_ratio,
            )

    def forward(self, x):
        if self.use_dual_pathway:

            def _block_forward(x):
                dual_output, _ = self.dual_pathway(self.dual_norm(x))
                x = x + dual_output
                x = x + self.ffn(self.ffn_norm(x))
                return x

        else:

            def _block_forward(x):
                return x + self.ffn(self.ffn_norm(x))

        if self.use_checkpoint and torch.is_grad_enabled():
            return checkpoint(_block_forward, x, use_reentrant=False)

        return _block_forward(x)


# =========================================================
# Pixel-wise downsample / upsample
# =========================================================
class PixelDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(4 * in_channels, out_channels, 1),
        )

    def forward(self, x):
        return self.body(x)


class PixelUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 4 * out_channels, 1),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


# =========================================================
# CoDe-SSM
# =========================================================
@ARCH_REGISTRY.register()
class CoDeSSM(nn.Module):
    def __init__(
        self,
        input_dim=3,
        output_dim=3,
        feature_dim=32,
        num_clusters=[16, 24, 32],
        heads=[1, 2, 4],
        num_blocks=[2, 4, 4],
        ffn_expansion_factor=2.66,
        use_checkpoint=True,
        hf_num_experts=4,
        hf_top_k=2,
        hf_expert_ratio=0.5,
        hf_router_hidden_ratio=0.125,
        pad_multiple=8,
    ):
        super().__init__()

        self.pad_multiple = pad_multiple

        moe_kwargs = dict(
            hf_num_experts=hf_num_experts,
            hf_top_k=hf_top_k,
            hf_expert_ratio=hf_expert_ratio,
            hf_router_hidden_ratio=hf_router_hidden_ratio,
        )

        self.input_projection = nn.Conv2d(input_dim, feature_dim, 3, 1, 1)
        self.initial_downsample = PixelDownsample(feature_dim, feature_dim)


        self.encoder_level1 = nn.Sequential(
            *[
                CoDeBlock(
                    use_dual_pathway=False,
                    feature_dim=feature_dim,
                    num_clusters=num_clusters[0],
                    heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    use_checkpoint=use_checkpoint,
                    **moe_kwargs,
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.downsample_1_to_2 = PixelDownsample(feature_dim, feature_dim * 2)

        self.encoder_level2 = nn.Sequential(
            *[
                CoDeBlock(
                    use_dual_pathway=False,
                    feature_dim=feature_dim * 2,
                    num_clusters=num_clusters[1],
                    heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    use_checkpoint=use_checkpoint,
                    **moe_kwargs,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.downsample_2_to_3 = PixelDownsample(feature_dim * 2, feature_dim * 4)

        self.encoder_level3 = nn.Sequential(
            *[
                CoDeBlock(
                    use_dual_pathway=False,
                    feature_dim=feature_dim * 4,
                    num_clusters=num_clusters[2],
                    heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    use_checkpoint=use_checkpoint,
                    **moe_kwargs,
                )
                for _ in range(num_blocks[2])
            ]
        )


        self.latent = nn.Sequential(
            *[
                CoDeBlock(
                    use_dual_pathway=True,
                    feature_dim=feature_dim * 4,
                    num_clusters=num_clusters[2],
                    heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    use_checkpoint=use_checkpoint,
                    **moe_kwargs,
                )
                for _ in range(num_blocks[2])
            ]
        )


        self.decoder_level3 = nn.Sequential(
            *[
                CoDeBlock(
                    use_dual_pathway=True,
                    feature_dim=feature_dim * 4,
                    num_clusters=num_clusters[2],
                    heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    use_checkpoint=use_checkpoint,
                    **moe_kwargs,
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.channel_reduction_level3 = nn.Conv2d(feature_dim * 8, feature_dim * 4, 1)
        self.upsample_3_to_2 = PixelUpsample(feature_dim * 4, feature_dim * 2)

        self.decoder_level2 = nn.Sequential(
            *[
                CoDeBlock(
                    use_dual_pathway=True,
                    feature_dim=feature_dim * 2,
                    num_clusters=num_clusters[1],
                    heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    use_checkpoint=use_checkpoint,
                    **moe_kwargs,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.channel_reduction_level2 = nn.Conv2d(feature_dim * 4, feature_dim * 2, 1)
        self.upsample_2_to_1 = PixelUpsample(feature_dim * 2, feature_dim)

        self.decoder_level1 = nn.Sequential(
            *[
                CoDeBlock(
                    use_dual_pathway=True,
                    feature_dim=feature_dim,
                    num_clusters=num_clusters[0],
                    heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    use_checkpoint=use_checkpoint,
                    **moe_kwargs,
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.channel_reduction_level1 = nn.Conv2d(feature_dim * 2, feature_dim, 1)
        self.final_upsample = PixelUpsample(feature_dim, feature_dim)

        self.refinement = nn.Sequential(
            *[
                CoDeBlock(
                    use_dual_pathway=False,
                    feature_dim=feature_dim,
                    num_clusters=num_clusters[0],
                    heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    use_checkpoint=use_checkpoint,
                    **moe_kwargs,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.output_projection = nn.Conv2d(feature_dim, output_dim, 3, 1, 1)
        self.global_residual_gate = nn.Parameter(torch.tensor(2.0))

        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _pad_input(self, x):
        h, w = x.shape[-2:]

        pad_h = (self.pad_multiple - h % self.pad_multiple) % self.pad_multiple
        pad_w = (self.pad_multiple - w % self.pad_multiple) % self.pad_multiple

        if pad_h == 0 and pad_w == 0:
            return x, h, w

        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), h, w

    def lhfm_balance_loss(self):
        losses = []

        for module in self.modules():
            if isinstance(module, LocalHighFrequencyModule):
                if module.last_lhfm_balance_loss is not None:
                    loss_value = module.last_lhfm_balance_loss
                    if torch.isfinite(loss_value):
                        losses.append(loss_value)

        if not losses:
            return None

        return torch.stack(losses).mean()

    def forward(self, x, return_aux=False):
        x, original_h, original_w = self._pad_input(x)

        identity = x

        feature = self.initial_downsample(self.input_projection(x))

        encoder_1 = self.encoder_level1(feature)
        encoder_2 = self.encoder_level2(self.downsample_1_to_2(encoder_1))
        encoder_3 = self.encoder_level3(self.downsample_2_to_3(encoder_2))

        latent_feature = self.latent(encoder_3)

        decoder_3 = self.decoder_level3(
            self.channel_reduction_level3(
                torch.cat([latent_feature, encoder_3], 1)
            )
        )

        decoder_2 = self.decoder_level2(
            self.channel_reduction_level2(
                torch.cat([self.upsample_3_to_2(decoder_3), encoder_2], 1)
            )
        )

        decoder_1 = self.decoder_level1(
            self.channel_reduction_level1(
                torch.cat([self.upsample_2_to_1(decoder_2), encoder_1], 1)
            )
        )

        decoder_1 = self.refinement(decoder_1)
        decoder_1 = self.final_upsample(decoder_1)

        output = self.output_projection(decoder_1)
        output = identity + torch.sigmoid(self.global_residual_gate) * output
        output = output[:, :, :original_h, :original_w]

        if return_aux:
            aux_loss = self.lhfm_balance_loss()
            return output, {
                "lhfm_balance_loss": aux_loss,
            }

        return output

    @torch.no_grad()
    def test(self, x):
        return self.forward(x)

    @torch.no_grad()
    def test_tile(self, x, tile_size=512, tile_pad=32):
        b, c, h, w = x.shape
        output = torch.zeros_like(x)

        for y0 in range(0, h, tile_size):
            for x0 in range(0, w, tile_size):
                y1 = min(y0 + tile_size, h)
                x1 = min(x0 + tile_size, w)

                y0_pad = max(y0 - tile_pad, 0)
                x0_pad = max(x0 - tile_pad, 0)
                y1_pad = min(y1 + tile_pad, h)
                x1_pad = min(x1 + tile_pad, w)

                tile = x[:, :, y0_pad:y1_pad, x0_pad:x1_pad]
                pred = self.forward(tile)

                output[:, :, y0:y1, x0:x1] = pred[
                    :,
                    :,
                    y0 - y0_pad : y1 - y0_pad,
                    x0 - x0_pad : x1 - x0_pad,
                ]

        return output