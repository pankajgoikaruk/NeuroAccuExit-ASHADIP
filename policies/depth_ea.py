# policies/depth_ea.py
import torch
import torch.nn.functional as F


@torch.no_grad()
def depth_ea_decide(
    logits_list,
    temps,
    ea_mode="logprob",
    ea_threshold=0.20,
    ea_min_exit=0,
    ea_stable_k=1,
    ea_flip_penalty=0.0,

    # Exit1 high-confidence override
    ea_exit1_conf_min=None,        # if None and ea_stable_k>1 -> auto uses 0.95
    ea_exit1_margin_mult=2.0,      # require margin >= mult * ea_threshold for Exit1
    ea_exit1_margin_min=0.0,       # optional absolute floor on Exit1 margin
):
    """
    Depth Evidence Accumulation across exits.

    NEW BEHAVIOR:
    - If ea_stable_k > 1, Exit1 would normally be impossible.
      We allow Exit1 only if it is VERY confident:
        conf >= ea_exit1_conf_min  AND  margin >= max(ea_exit1_margin_min, ea_exit1_margin_mult * ea_threshold)

    NEW OUTPUT:
    - logp_taken: (B,C) log-probabilities after *accumulation* at the exit where each sample stopped.
                 (This is the posterior you should accumulate over time for clip-sequential inference.)
    - logp_final: (B,C) final accumulated posterior at last exit.
    """
    K = len(logits_list)
    B, C = logits_list[0].shape
    device = logits_list[0].device

    if ea_exit1_conf_min is None and ea_stable_k > 1:
        ea_exit1_conf_min = 0.95

    # Evidence accumulator in class-space
    S = torch.zeros((B, C), device=device)

    decided = torch.zeros(B, dtype=torch.bool, device=device)
    taken = torch.full((B,), K - 1, dtype=torch.long, device=device)
    pred_taken = torch.zeros(B, dtype=torch.long, device=device)
    margin_taken = torch.zeros(B, dtype=torch.float, device=device)

    # NEW: store accumulated posterior at the moment of exit
    logp_taken = torch.zeros((B, C), dtype=torch.float, device=device)

    prev_pred = None
    stable = torch.zeros(B, dtype=torch.long, device=device)
    flip_count = torch.zeros(B, dtype=torch.long, device=device)

    last_margin = torch.zeros(B, dtype=torch.float, device=device)
    last_pred = torch.zeros(B, dtype=torch.long, device=device)
    last_logp = torch.zeros((B, C), dtype=torch.float, device=device)

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
            S = S + F.log_softmax(lg, dim=1)

        # Posterior after accumulation
        p = F.softmax(S, dim=1)
        logp = F.log_softmax(S, dim=1)  # <-- this is the accumulated posterior in log-space

        top2 = torch.topk(p, 2, dim=1).values
        conf = top2[:, 0]
        margin = top2[:, 0] - top2[:, 1]
        pred = torch.argmax(p, dim=1)

        # Track stability / flips
        if prev_pred is None:
            stable[:] = 1
        else:
            same = (pred == prev_pred)
            flip_count += (~same).long()
            stable = torch.where(same, stable + 1, torch.ones_like(stable))
            if ea_flip_penalty > 0.0:
                margin = margin - (ea_flip_penalty * (~same).float())

        # Save last seen
        last_margin = margin
        last_pred = pred
        last_logp = logp

        # Exit1 override gate
        exit1_ok = torch.zeros(B, dtype=torch.bool, device=device)
        if k == 0 and ea_exit1_conf_min is not None and ea_min_exit <= 0:
            req_margin = max(float(ea_exit1_margin_min), float(ea_exit1_margin_mult) * float(ea_threshold))
            exit1_ok = (conf >= float(ea_exit1_conf_min)) & (margin >= req_margin)

        stable_ok = (stable >= ea_stable_k) | exit1_ok
        eligible = (k >= ea_min_exit) & stable_ok & (margin >= ea_threshold)

        newly = eligible & (~decided)
        if newly.any():
            taken[newly] = k
            pred_taken[newly] = pred[newly]
            margin_taken[newly] = margin[newly]
            logp_taken[newly] = logp[newly]   # <-- NEW: store accumulated posterior at exit moment
            decided[newly] = True

        prev_pred = pred

    pred_final = last_pred.clone()

    never = ~decided
    if never.any():
        pred_taken[never] = pred_final[never]
        margin_taken[never] = last_margin[never]
        logp_taken[never] = last_logp[never]  # <-- use final accumulated posterior if never exited

    return {
        "taken": taken,
        "pred_taken": pred_taken,
        "pred_final": pred_final,
        "flip_count": flip_count,
        "margin_taken": margin_taken,

        # NEW:
        "logp_taken": logp_taken,
        "logp_final": last_logp,
    }
