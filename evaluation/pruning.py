"""Routed-expert pruning for explicit Linear and stacked Qwen3 expert layouts.

Calibration collects dense-model activation energies, then applies masks. This
is a reconstructed workflow, not a claim of bitwise identity to archived runs.
"""
from dataclasses import dataclass

import torch
from torch import nn

from evaluation.wanda.lib.hb_nm import (
    HybridBlockSparseConfig, build_hybrid_block_sparse_prune_mask, validate_hb_nm_shape,
)


@dataclass(frozen=True)
class Recipe:
    pattern: str = "hibnm"
    method: str = "wanda"
    sparsity: float = 0.25
    block_h: int = 64
    block_w: int = 64
    block_n: int = 1
    block_m: int = 2
    score: str = "max_row_squared"

    def validate(self):
        if self.method not in {"wanda", "magnitude"}:
            raise ValueError("method must be wanda or magnitude")
        if self.pattern not in {"dense", "hibnm", "unstructured", "2:4", "2:8"}:
            raise ValueError("unsupported pattern")
        if not 0 <= self.sparsity <= 1:
            raise ValueError("sparsity must be in [0, 1]")
        if self.pattern == "hibnm":
            if min(self.block_h, self.block_w, self.block_n, self.block_m) <= 0:
                raise ValueError("block dimensions/counts must be positive")
            if self.block_n > self.block_m or self.block_w % 4:
                raise ValueError("invalid HiBNM block configuration")
            expected = self.block_n / (2 * self.block_m)
        else:
            expected = {"dense": 0, "2:4": .5, "2:8": .75}.get(self.pattern)
        if expected is not None and abs(self.sparsity - expected) > 1e-9:
            raise ValueError(f"{self.pattern} requires sparsity={expected}")
        if self.score not in {"sum", "squared_sum", "max_row_squared"}:
            raise ValueError("unknown block score")


def make_mask(importance, recipe):
    """True means removed; 2:8 retains exactly TWO, not six, values per octet."""
    recipe.validate()
    if importance.ndim != 2 or not torch.isfinite(importance).all():
        raise ValueError("importance must be a finite 2-D matrix")
    if recipe.pattern == "hibnm":
        return build_hybrid_block_sparse_prune_mask(importance, HybridBlockSparseConfig(
            recipe.block_h, recipe.block_w, recipe.block_n, recipe.block_m, recipe.score))
    mask = torch.zeros_like(importance, dtype=torch.bool)
    if recipe.pattern == "dense":
        return mask
    if recipe.pattern in {"2:4", "2:8"}:
        width = int(recipe.pattern.split(":")[1])
        if importance.shape[1] % width:
            raise ValueError("input dimension must be divisible by N:M group width")
        groups = importance.reshape(importance.shape[0], -1, width)
        order = groups.argsort(dim=-1, stable=True)
        return torch.zeros_like(groups, dtype=torch.bool).scatter_(
            -1, order[..., :width - 2], True).reshape_as(importance)
    count = int(importance.shape[1] * recipe.sparsity)
    return mask.scatter_(1, importance.argsort(dim=1, stable=True)[:, :count], True)


@dataclass
class Target:
    name: str
    weight: torch.Tensor
    energy: torch.Tensor | None = None
    tokens: int = 0

    def observe(self, x):
        x = x.detach().reshape(-1, x.shape[-1]).float()
        energy = x.square().sum(0)
        self.energy = energy if self.energy is None else self.energy + energy
        self.tokens += x.shape[0]


def expert_targets(model, collect=False):
    """Return weight views and removable hooks; reject unknown layouts."""
    targets, handles = [], []
    try:
        for name, module in model.named_modules():
            if not name.endswith("mlp.experts"):
                continue
            if isinstance(module, nn.ModuleList):
                for index, expert in enumerate(module):
                    for projection in ("gate_proj", "up_proj", "down_proj"):
                        linear = getattr(expert, projection, None)
                        if not isinstance(linear, nn.Linear):
                            raise ValueError(f"unsupported expert projection: {name}.{index}.{projection}")
                        target = Target(f"{name}.{index}.{projection}", linear.weight)
                        targets.append(target)
                        if collect:
                            handles.append(linear.register_forward_pre_hook(
                                lambda mod, args, t=target: t.observe(args[0])))
            elif all(isinstance(getattr(module, p, None), nn.Parameter)
                     for p in ("gate_up_proj", "down_proj")):
                gate_up, down = module.gate_up_proj, module.down_proj
                if (gate_up.ndim != 3 or down.ndim != 3 or
                    gate_up.shape[0] != down.shape[0] or
                    gate_up.shape[1] != 2 * down.shape[2] or
                    gate_up.shape[2] != down.shape[1]):
                    raise ValueError(f"unsupported stacked expert axes: {name}")
                pairs = []
                for index in range(gate_up.shape[0]):
                    gate, up = gate_up[index].chunk(2, dim=0)
                    group = [Target(f"{name}.{index}.{p}", w) for p, w in
                             (("gate_proj", gate), ("up_proj", up), ("down_proj", down[index]))]
                    targets.extend(group)
                    pairs.append(group)
                if collect:
                    def observe(mod, args, kwargs, pairs=pairs):
                        hidden = args[0] if args else kwargs["hidden_states"]
                        indices = args[1] if len(args) > 1 else kwargs["top_k_index"]
                        hidden = hidden.reshape(-1, hidden.shape[-1])
                        if indices.ndim != 2 or indices.shape[0] != hidden.shape[0]:
                            raise ValueError("unsupported stacked expert routing signature")
                        for index, (gate, up, down) in enumerate(pairs):
                            rows = torch.where(indices == index)[0]
                            if rows.numel():
                                x = hidden[rows]
                                gate.observe(x)
                                up.observe(x)
                                intermediate = mod.act_fn(nn.functional.linear(x, gate.weight))
                                intermediate = intermediate * nn.functional.linear(x, up.weight)
                                down.observe(intermediate)
                    handles.append(module.register_forward_pre_hook(observe, with_kwargs=True))
            else:
                raise ValueError(f"unsupported routed expert container: {name}")
        if not targets:
            raise ValueError("no routed expert weights found; refusing a no-op prune")
        return targets, handles
    except Exception:
        for handle in handles:
            handle.remove()
        raise


@torch.no_grad()
def prune_model(model, recipe, calibration=(), uncalibrated="error"):
    recipe.validate()
    if uncalibrated not in {"error", "magnitude"}:
        raise ValueError("unknown uncalibrated-expert policy")
    collect = recipe.method == "wanda" and recipe.pattern != "dense"
    targets, handles = expert_targets(model, collect)
    use_cache = model.config.use_cache
    try:
        model.config.use_cache = False
        if collect:
            for batch in calibration:
                model(**batch)
    finally:
        model.config.use_cache = use_cache
        for handle in handles:
            handle.remove()
    # Check every target before modifying any weights.
    missing = [t.name for t in targets if collect and not t.tokens]
    if missing and uncalibrated == "error":
        raise ValueError(f"{len(missing)} projections received no calibration tokens: {missing[:3]}")
    for target in targets:
        if recipe.pattern == "hibnm":
            validate_hb_nm_shape(target.weight.shape, HybridBlockSparseConfig(
                recipe.block_h, recipe.block_w, recipe.block_n, recipe.block_m, recipe.score))
        if recipe.pattern in {"2:4", "2:8"} and target.weight.shape[1] % int(recipe.pattern[-1]):
            raise ValueError("input dimension must be divisible by N:M group width")
        if not torch.isfinite(target.weight).all() or (target.energy is not None and not torch.isfinite(target.energy).all()):
            raise ValueError(f"nonfinite weights or activation energy: {target.name}")
    records = []
    for target in targets:
        importance = target.weight.detach().abs().float()
        if collect and target.tokens:
            importance *= target.energy.sqrt().unsqueeze(0)
        mask = make_mask(importance, recipe)
        target.weight.masked_fill_(mask, 0)
        records.append({"name": target.name, "shape": list(target.weight.shape),
                        "parameters": target.weight.numel(), "masked": int(mask.sum()),
                        "zeros": int((target.weight == 0).sum()), "calibration_tokens": target.tokens,
                        "magnitude_fallback": collect and not target.tokens})
    return records
