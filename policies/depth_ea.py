from __future__ import annotations

from typing import List, Optional, Sequence, Dict, Any
import torch
import torch.nn.functional as F


@torch.no_grad()
def depth_ea_decide(
    logits_list: List[torch.Tensor],
    temps: Optional[Sequence[float]],
    ea_mode: str = "logprob",          # "logprob" (recommended) or "logits"
    ea_threshold: float = 0.20,
    ea_min_exit: int = 0,
    ea_stable_k: int = 1,
    ea_flip_penalty: float = 0.0,
    # Exit1 high-confidence override
    ea_exit1_conf_min: Optional[float] = None,   # if None and ea_stable_k>1 -> default 0.95
    ea_exit1_margin_mult: float = 2.0,           # require margin >= mult * ea_threshold for Exit1 override
    ea_exit1_margin_min: float = 0.0,            # optional absolute floor on Exit1 margin override
    # Exit2 strict confirmation override
    ea_exit2_conf_min: float = 0.99,
    ea_exit2_margin_min: float = 0.45,
    ea_exit2_conf_gain_min: float = 0.05,
    ea_exit2_margin_gain_min: float = 0.03,
    ea_exit2_require_agree: bool = True,
) -> Dict[str, Any]:
    """
    Depth Evidence Accumulation across exits (generic K exits).

    Returns dict with:
      - taken: (B,) long in [0..K-1]
      - pred_taken: (B,) long
      - pred_final: (B,) long
      - flip_count: (B,) long
      - logp_taken: (B,C) log-probabilities at chosen exit AFTER accumulation+normalization
      - logp_final: (B,C) log-probabilities at final exit AFTER accumulation+normalization
    """
    if not logits_list:
        raise ValueError("logits_list is empty.")

    K = len(logits_list)
    B, C = logits_list[0].shape
    dev = logits_list[0].device

    # Validate shapes across exits
    for i, lg in enumerate(logits_list):
        if lg.dim() != 2:
            raise ValueError(f"logits_list[{i}] must be 2D (B,C), got shape {tuple(lg.shape)}")
        if lg.shape[0] != B or lg.shape[1] != C:
            raise ValueError(
                f"All exits must have same shape. logits_list[0]={tuple(logits_list[0].shape)} "
                f"but logits_list[{i}]={tuple(lg.shape)}"
            )
        if lg.device != dev:
            raise ValueError("All exit logits must be on the same device.")

    if C < 2:
        raise ValueError(f"Need at least 2 classes for margin/top2. Got C={C}")

    # temps handling (pad/truncate to K)
    if temps is None:
        temps = [1.0] * K
    temps = [max(float(t), 1e-3) for t in temps]
    if len(temps) < K:
        temps = temps + [temps[-1]] * (K - len(temps))
    elif len(temps) > K:
        temps = temps[:K]

    if ea_mode not in ("logprob", "logits"):
        raise ValueError(f"ea_mode must be 'logprob' or 'logits'. Got: {ea_mode}")

    # evidence per exit
    evid: List[torch.Tensor] = []
    for k in range(K):
        lg = logits_list[k].float() / temps[k]
        if ea_mode == "logits":
            evid.append(lg)  # accumulate logits
        else:
            evid.append(F.log_softmax(lg, dim=1))  # accumulate log-probs

    # cumulative evidence
    cum: List[torch.Tensor] = []
    acc = torch.zeros_like(evid[0])
    for k in range(K):
        acc = acc + evid[k]
        cum.append(acc)

    # normalized log-posteriors at each depth
    logp_depth = [F.log_softmax(c, dim=1) for c in cum]  # each (B,C)

    # per-depth preds/margins/conf
    preds: List[torch.Tensor] = []
    confs: List[torch.Tensor] = []
    margins: List[torch.Tensor] = []
    for k in range(K):
        p = torch.exp(logp_depth[k])  # (B,C)
        top2 = torch.topk(p, k=2, dim=1).values  # (B,2)
        conf = top2[:, 0]
        margin = top2[:, 0] - top2[:, 1]
        pred = torch.argmax(p, dim=1)
        preds.append(pred)
        confs.append(conf)
        margins.append(margin)

    # scan exits left-to-right
    taken = torch.full((B,), K - 1, dtype=torch.long, device=dev)
    pred_taken = preds[-1].clone()
    pred_final = preds[-1].clone()

    undecided = torch.ones((B,), dtype=torch.bool, device=dev)
    stable = torch.ones((B,), dtype=torch.long, device=dev)
    flips = torch.zeros((B,), dtype=torch.long, device=dev)

    for k in range(K):
        if k > 0:
            same = preds[k].eq(preds[k - 1])
            stable = torch.where(same, stable + 1, torch.ones_like(stable))
            flips = torch.where(same, flips, flips + 1)

        eff_thr = ea_threshold + ea_flip_penalty * flips.float()
        ok_margin = margins[k].ge(eff_thr)
        ok_min_exit = torch.full((B,), k >= int(ea_min_exit), dtype=torch.bool, device=dev)
        ok_stable = stable.ge(int(ea_stable_k))

        # normal EA gating
        can_take = undecided & ok_min_exit & ok_margin & ok_stable

        # Exit1 override when stable_k > 1 (otherwise exit1 never becomes stable)
        if k == 0 and int(ea_stable_k) > 1:
            conf_min = 0.95 if ea_exit1_conf_min is None else float(ea_exit1_conf_min)
            override_margin = max(float(ea_exit1_margin_min), float(ea_exit1_margin_mult) * float(ea_threshold))
            can_take_exit1 = undecided & confs[0].ge(conf_min) & margins[0].ge(override_margin)
            can_take = can_take | can_take_exit1

        # Exit2 strict confirmation override
        # This is a rare exception that can bypass ea_min_exit=2,
        # but only when exit2 is clearly stronger than exit1.
        if k == 1:
            conf1 = confs[0]
            margin1 = margins[0]
            pred1 = preds[0]

            conf2 = confs[1]
            margin2 = margins[1]
            pred2 = preds[1]

            conf_gain12 = conf2 - conf1
            margin_gain12 = margin2 - margin1
            agree12 = pred1.eq(pred2)

            can_take_exit2 = (
                undecided
                & conf2.ge(float(ea_exit2_conf_min))
                & margin2.ge(float(ea_exit2_margin_min))
                & conf_gain12.ge(float(ea_exit2_conf_gain_min))
                & margin_gain12.ge(float(ea_exit2_margin_gain_min))
            )

            if ea_exit2_require_agree:
                can_take_exit2 = can_take_exit2 & agree12

            can_take = can_take | can_take_exit2

        if can_take.any():
            idx = can_take.nonzero(as_tuple=False).squeeze(1)
            taken[idx] = k
            pred_taken[idx] = preds[k][idx]
            undecided[idx] = False

        if not undecided.any():
            break

    # gather logp_taken from stacked [K,B,C]
    stack = torch.stack(logp_depth, dim=0)  # (K,B,C)
    b_idx = torch.arange(B, device=dev)
    logp_taken = stack[taken, b_idx, :]  # (B,C)
    logp_final = logp_depth[-1]

    return {
        "taken": taken,
        "pred_taken": pred_taken,
        "pred_final": pred_final,
        "flip_count": flips,
        "logp_taken": logp_taken,
        "logp_final": logp_final,
    }