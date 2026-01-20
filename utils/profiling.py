# utils/profiling.py

import time, torch


@torch.no_grad()
def measure_latency_ms(model, batch, n_warm=5, n_iter=20, device='cpu'):
    model.eval()
    batch = batch.to(device)
    for _ in range(n_warm):
        _ = model(batch)
    torch.cuda.synchronize() if device.startswith('cuda') else None
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = model(batch)
    torch.cuda.synchronize() if device.startswith('cuda') else None
    dt = (time.perf_counter()-t0)/n_iter*1000
    return dt



def conv2d_flops(
    h, w,
    in_ch=None, out_ch=None,
    k=3, stride=1, padding=1,
    cin=None, cout=None
):
    """
    Backward-compatible conv2d FLOPs:
    - Old style: conv2d_flops(h, w, in_ch=..., out_ch=...)
    - New style: conv2d_flops(h, w, cin=..., cout=...)
    """
    if in_ch is None:
        in_ch = cin
    if out_ch is None:
        out_ch = cout

    if in_ch is None or out_ch is None:
        raise TypeError("conv2d_flops requires in_ch/out_ch or cin/cout")

    # MACs = H_out*W_out*out_ch*(in_ch*k*k); FLOPs ≈ 2*MACs (mul+add)
    h_out = (h + 2 * padding - k) // stride + 1
    w_out = (w + 2 * padding - k) // stride + 1
    macs = h_out * w_out * out_ch * (in_ch * k * k)
    return 2 * macs, h_out, w_out


def estimate_flops_tiny_audiocnn(n_mels=64, frames=100, num_classes=2):
    """
    Estimate FLOPs up to each exit for the 5-block TinyAudioCNN (Option A: exits 1/3/5).

    Assumptions (match adapters/audio_adapter.py):
      - block1: Conv(1->16,k3,p1) + MaxPool(2x2)
      - block2: Conv(16->24,k3,p1) + MaxPool(2x2)
      - block3: Conv(24->32,k3,p1)
      - block4: Conv(32->48,k3,p1)
      - block5: Conv(48->64,k3,p1) + AdaptiveAvgPool(1,1)

    Exit heads (ExitNet):
      - exit1 head input dim = 16  (after block1 tap)
      - exit2 head input dim = 32  (after block3 tap)
      - exit3 head input dim = 64  (final embedding after block5)

    Note:
      Conv FLOPs dominate; linear heads are tiny, but we include them for completeness.
    """
    flops = {}
    total = 0

    # Start with full-resolution mel/time
    H, W = n_mels, frames  # treat as "image" height/width

    # ---------------- block1 conv ----------------
    f1, h1, w1 = conv2d_flops(H, W, cin=1, cout=16, k=3, stride=1, padding=1)
    total += f1

    # MaxPool 2x2: halves spatial dims
    H1, W1 = h1 // 2, w1 // 2

    # exit1 head: 16 -> C (approx 2*in*out for multiply+add)
    flops["exit1"] = total + 2 * (16 * num_classes)

    # ---------------- block2 conv ----------------
    f2, h2, w2 = conv2d_flops(H1, W1, cin=16, cout=24, k=3, stride=1, padding=1)
    total += f2

    # MaxPool 2x2 again
    H2, W2 = h2 // 2, w2 // 2

    # ---------------- block3 conv ----------------
    f3, h3, w3 = conv2d_flops(H2, W2, cin=24, cout=32, k=3, stride=1, padding=1)
    total += f3

    # exit2 head: 32 -> C
    flops["exit2"] = total + 2 * (32 * num_classes)

    # ---------------- block4 conv ----------------
    f4, h4, w4 = conv2d_flops(h3, w3, cin=32, cout=48, k=3, stride=1, padding=1)
    total += f4

    # ---------------- block5 conv ----------------
    f5, h5, w5 = conv2d_flops(h4, w4, cin=48, cout=64, k=3, stride=1, padding=1)
    total += f5

    # AdaptiveAvgPool -> (1,1) (ignore FLOPs; negligible compared to conv)
    # exit3 head: 64 -> C
    flops["exit3"] = total + 2 * (64 * num_classes)

    return flops


