# policies/depth_ea.py
import torch
import torch.nn.functional as F


@torch.no_grad()
def depth_ea_decide(
    logits_list,
    temps,
    ea_mode="logprob",         # "logprob" (recommended) or "logits"
    ea_threshold=0.20,         # margin threshold after accumulation (probability margin)
    ea_min_exit=0,             # 0/1/2 -> earliest exit allowed
    ea_stable_k=1,             # require same predicted class for last K exits
    ea_flip_penalty=0.0,       # subtract penalty if pred flips between exits
):
    """
    Depth Evidence Accumulation across exits.

    logits_list: [logits1, logits2, logits3], each (B,C)
    temps: list of 3 temps (or None) used for scaling (per-exit)
    Decision is made at exit k if:
      - k >= ea_min_exit
      - predicted class stayed same for >= ea_stable_k consecutive exits
      - margin(top1 - top2) of accumulated posterior >= ea_threshold
      - optional flip penalty reduces margin when prediction changes

    Returns dict with:
      taken (B,), pred_taken (B,), pred_final (B,), flip_count (B,), margin_taken (B,)
    """
    K = len(logits_list)
    B, C = logits_list[0].shape
    device = logits_list[0].device

    # Evidence accumulator in class-space
    S = torch.zeros((B, C), device=device)

    decided = torch.zeros(B, dtype=torch.bool, device=device)
    taken = torch.full((B,), K - 1, dtype=torch.long, device=device)
    pred_taken = torch.zeros(B, dtype=torch.long, device=device)
    margin_taken = torch.zeros(B, dtype=torch.float, device=device)

    prev_pred = None
    stable = torch.zeros(B, dtype=torch.long, device=device)
    flip_count = torch.zeros(B, dtype=torch.long, device=device)

    last_margin = torch.zeros(B, dtype=torch.float, device=device)
    last_pred = torch.zeros(B, dtype=torch.long, device=device)

    for k in range(K):
        lg = logits_list[k]

        # Temperature scaling
        if temps is not None:
            t = max(float(temps[k]), 1e-3)
            lg = lg / t

        # Accumulate evidence
        if ea_mode == "logits":
            S = S + lg
        else:
            # default: log-prob accumulation (stable)
            S = S + F.log_softmax(lg, dim=1)

        # Posterior after accumulation
        p = F.softmax(S, dim=1)
        top2 = torch.topk(p, 2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        pred = torch.argmax(p, dim=1)

        # Track stability / flips
        if prev_pred is None:
            stable[:] = 1  # first exit counts as "1 consecutive"
        else:
            same = (pred == prev_pred)
            flip_count += (~same).long()
            stable = torch.where(same, stable + 1, torch.ones_like(stable))

            if ea_flip_penalty > 0.0:
                margin = margin - (ea_flip_penalty * (~same).float())

        # Save last seen (for never-decide case)
        last_margin = margin
        last_pred = pred

        # Eligibility check
        eligible = (k >= ea_min_exit) & (stable >= ea_stable_k) & (margin >= ea_threshold)

        newly = eligible & (~decided)
        if newly.any():
            taken[newly] = k
            pred_taken[newly] = pred[newly]
            margin_taken[newly] = margin[newly]
            decided[newly] = True

        prev_pred = pred

    # Final prediction after the last exit
    pred_final = last_pred.clone()

    # If never decided early, fall back to final
    never = ~decided
    if never.any():
        pred_taken[never] = pred_final[never]
        margin_taken[never] = last_margin[never]

    return {
        "taken": taken,                 # 0/1/2
        "pred_taken": pred_taken,
        "pred_final": pred_final,
        "flip_count": flip_count,
        "margin_taken": margin_taken,
    }
