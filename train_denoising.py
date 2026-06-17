#!/usr/bin/env python3
"""
train_denoising.py — Denoising-based training for unit-distance graph embedding.

THE KEY IDEA (why this works when cold-start training doesn't):
  During training: input = x_0 + σ·ε,  target = x_0
  At σ_max (≈5× embedding radius): input is pure random noise → identical to test-time cold start.
  At σ_min (≈0.02): input is nearly clean → just fine refinement.
  Training on ALL σ simultaneously teaches the model to embed from scratch AND to refine.

The current train_udg_embedder.py trains on x_0 + fixed large noise, so the model sees
the library only as noisy blobs to denoise at one scale. This version:
  1. Samples σ uniformly on a log scale every batch — full spectrum coverage.
  2. Adds a Procrustes-aligned POSITION LOSS alongside edge-length loss — direct supervision
     from the library's geometric knowledge, not just constraint satisfaction.
  3. Uses iterative refinement (T recycling passes, shared weights) for test-time inference.
  4. Optionally runs a curriculum: ring 1 → 2 → ... → 5, with ring-r models
     warm-starting from ring-(r-1) checkpoints.

USAGE (single stage):
  python train_denoising.py --lib udg_library --rings 4 --epochs 300

USAGE (curriculum, ring 1 → 5):
  python train_denoising.py --lib udg_library --curriculum --epochs-per-stage 150

USAGE (cold-start evaluation with T iterative passes):
  python train_denoising.py --lib udg_library --eval-only --ckpt checkpoints/best.pt --T 8
"""

import argparse, glob, json, math, os, sys, time
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    sys.exit("pip install torch")

# ── Procrustes alignment ──────────────────────────────────────────────────────
def procrustes_align(pred, target):
    """
    Rotate pred to best match target (no scaling, no reflection).
    Returns aligned_pred and the rotation matrix R.
    Both inputs: (V, d), centered.
    """
    pred_c = pred - pred.mean(0, keepdim=True)
    tgt_c  = target - target.mean(0, keepdim=True)
    # Optimal rotation: SVD of cross-covariance
    H = pred_c.T @ tgt_c          # (d, d)
    try:
        U, S, Vt = torch.linalg.svd(H)
    except Exception:
        return pred_c, torch.eye(pred.shape[1], device=pred.device)
    R = Vt.T @ U.T
    # Ensure proper rotation (det = +1)
    if torch.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return pred_c @ R.T, R


# ── EGNN (same architecture as train_udg_embedder.py) ────────────────────────
class EGNNLayer(nn.Module):
    def __init__(self, h_dim, noise_emb_dim=16):
        super().__init__()
        # Edge: (h_i, h_j, ||xi-xj||², L_ij, L_ij², σ_emb) → message
        self.edge_net = nn.Sequential(
            nn.Linear(2*h_dim + 3 + noise_emb_dim, h_dim), nn.SiLU(),
            nn.Linear(h_dim, h_dim), nn.SiLU())
        self.coord_net = nn.Sequential(
            nn.Linear(h_dim, h_dim//2), nn.SiLU(),
            nn.Linear(h_dim//2, 1))
        self.node_net = nn.Sequential(
            nn.Linear(2*h_dim, h_dim), nn.SiLU(),
            nn.Linear(h_dim, h_dim))

    def forward(self, h, x, edge_src, edge_dst, L, sigma_emb):
        # sigma_emb: (E, noise_emb_dim) — noise level broadcast to edges
        dv = x[edge_src] - x[edge_dst]
        d2 = (dv*dv).sum(-1, keepdim=True)
        e_feat = torch.cat([h[edge_src], h[edge_dst], d2,
                            L.unsqueeze(-1), (L*L).unsqueeze(-1),
                            sigma_emb], dim=-1)
        msg = self.edge_net(e_feat)
        w = self.coord_net(msg)
        delta_x = torch.zeros_like(x)
        delta_x.scatter_add_(0, edge_src.unsqueeze(-1).expand_as(dv), w*dv)
        x = x + delta_x / (h.shape[0]+1e-6)
        agg = torch.zeros_like(h)
        agg.scatter_add_(0, edge_src.unsqueeze(-1).expand_as(msg), msg)
        h = h + self.node_net(torch.cat([h, agg], dim=-1))
        return h, x


class DenoisingEmbedder(nn.Module):
    """
    EGNN that takes (noisy positions, graph, σ) and predicts clean positions x_0.
    Noise level σ is embedded via sinusoidal encoding and conditioned at every layer,
    so the model knows "how much to denoise." This is the key difference from the
    previous architecture, which was noise-level-blind.
    """
    def __init__(self, h_dim=64, n_layers=6, sigma_min=0.02, sigma_max=5.25,
                 noise_emb_dim=16):
        super().__init__()
        self.h_dim = h_dim
        self.noise_emb_dim = noise_emb_dim
        self.log_sigma_min = math.log(sigma_min)
        self.log_sigma_max = math.log(sigma_max)
        self.node_enc = nn.Linear(1, h_dim)
        self.layers = nn.ModuleList(
            [EGNNLayer(h_dim, noise_emb_dim) for _ in range(n_layers)])

    def sigma_embedding(self, sigma, n_edges, device):
        """Sinusoidal noise-level embedding, broadcast to all edges."""
        # log-normalise σ to [0,1]
        s = (math.log(float(sigma)) - self.log_sigma_min) / (
            self.log_sigma_max - self.log_sigma_min)
        s = max(0.0, min(1.0, s))
        freqs = torch.arange(self.noise_emb_dim//2, device=device).float()
        freqs = freqs / (self.noise_emb_dim//2)
        freqs = 1.0 / (10000**freqs)
        emb = torch.cat([torch.sin(s*freqs), torch.cos(s*freqs)])  # (noise_emb_dim,)
        return emb.unsqueeze(0).expand(n_edges, -1)                # (E, noise_emb_dim)

    def forward(self, x_noisy, edge_src, edge_dst, L, sigma, T_refine=1):
        """
        x_noisy: (V, d)
        sigma: float — noise level used during this forward pass
        T_refine: number of recycling passes (shared weights)
                  T=1 at training time, T=4-8 at test time for cold start
        Returns predicted clean positions x_0_hat: (V, d)
        """
        deg = torch.bincount(edge_src, minlength=x_noisy.shape[0]).float()
        h0  = self.node_enc(torch.log1p(deg).unsqueeze(-1))
        sigma_emb = self.sigma_embedding(sigma, edge_src.shape[0], x_noisy.device)
        x = x_noisy - x_noisy.mean(0)   # center
        for _ in range(T_refine):
            h = h0.clone()
            for layer in self.layers:
                h, x = layer(h, x, edge_src, edge_dst, L, sigma_emb)
        return x


# ── Losses ────────────────────────────────────────────────────────────────────
def edge_length_loss(pos, src, dst, L):
    d = (pos[src] - pos[dst]).norm(dim=-1)
    return ((d - L)**2).mean()

def position_loss(pred, target_gt):
    """
    Procrustes-aligned MSE position loss.
    Directly supervises the model toward the library geometry.
    """
    tgt_c = target_gt - target_gt.mean(0)
    aligned, _ = procrustes_align(pred, tgt_c)
    return ((aligned - tgt_c)**2).mean()

def combined_loss(pred, target_gt, src, dst, L, pos_weight=1.0, edge_weight=0.5):
    pl = position_loss(pred, target_gt)
    el = edge_length_loss(pred, src, dst, L)
    return pos_weight*pl + edge_weight*el, pl, el


# ── Dataset ───────────────────────────────────────────────────────────────────
class UDGDataset(torch.utils.data.Dataset):
    def __init__(self, npz_files, val=False, val_frac=0.15, seed=0):
        by_base = {}
        for f in npz_files:
            base = '_'.join(os.path.basename(f).split('_')[:5])
            by_base.setdefault(base, []).append(f)
        bases = sorted(by_base)
        rng = np.random.default_rng(seed)
        rng.shuffle(bases)
        n_val = max(1, int(len(bases)*val_frac))
        val_bases = set(bases[:n_val])
        chosen = {b for b in bases if (b in val_bases)==val}
        self.files = [f for b in chosen for f in by_base[b]]
        tag = 'Val' if val else 'Train'
        print(f"{tag}: {len(self.files)} instances from {len(chosen)} surfaces")

    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        d = np.load(self.files[i])
        return (torch.tensor(d['positions'], dtype=torch.float32),
                torch.tensor(d['edges'],    dtype=torch.long),
                torch.tensor(d['lengths'],  dtype=torch.float32))


# ── Training ──────────────────────────────────────────────────────────────────
def train_stage(model, files, args, stage_name, prev_ckpt=None):
    device = torch.device(args.device)
    if prev_ckpt and os.path.exists(prev_ckpt):
        ck = torch.load(prev_ckpt, map_location=device)
        model.load_state_dict(ck['model'])
        print(f"Warm-started from {prev_ckpt}")

    model = model.to(device)
    train_set = UDGDataset(files, val=False, seed=args.seed)
    val_set   = UDGDataset(files, val=True,  seed=args.seed)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr*0.05)

    rng = np.random.default_rng(args.seed)
    sigma_min, sigma_max = args.sigma_min, args.sigma_max
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_val = float('inf')
    log_path = os.path.join(args.ckpt_dir, f'{stage_name}_log.jsonl')

    for epoch in range(1, args.epochs+1):
        model.train()
        rng.shuffle(train_set.files)
        t_loss = 0.0; n = 0
        for fname in train_set.files:
            d = np.load(fname)
            x0   = torch.tensor(d['positions'], dtype=torch.float32).to(device)
            edges= torch.tensor(d['edges'],     dtype=torch.long).to(device)
            L    = torch.tensor(d['lengths'],   dtype=torch.float32).to(device)
            src, dst = edges[:,0], edges[:,1]

            # Sample σ on log-scale — full spectrum every epoch
            log_s = rng.uniform(math.log(sigma_min), math.log(sigma_max))
            sigma = math.exp(log_s)

            # Add noise to ground truth to get training input
            noise = torch.randn_like(x0) * sigma
            x_noisy = x0 + noise

            pred = model(x_noisy, src, dst, L, sigma, T_refine=1)
            loss, pl, el = combined_loss(pred, x0, src, dst, L,
                                         args.pos_weight, args.edge_weight)
            opt.zero_grad(); loss.backward(); opt.step()
            t_loss += loss.item(); n += 1
        sched.step()

        if epoch % args.eval_every != 0:
            continue

        # ── validate ──
        model.eval()
        v_pos, v_edge, v_succ = [], [], []
        with torch.no_grad():
            for fname in val_set.files[:min(30, len(val_set.files))]:
                d = np.load(fname)
                x0   = torch.tensor(d['positions'], dtype=torch.float32).to(device)
                edges= torch.tensor(d['edges'],     dtype=torch.long).to(device)
                L    = torch.tensor(d['lengths'],   dtype=torch.float32).to(device)
                src, dst = edges[:,0], edges[:,1]
                # Cold-start eval: pure random init at σ_max, then T_refine passes
                x_cold = torch.randn_like(x0) * sigma_max
                pred = model(x_cold, src, dst, L, sigma_max, T_refine=args.T_eval)
                _, pl, el = combined_loss(pred, x0, src, dst, L)
                err = (pred[src]-pred[dst]).norm(dim=-1) - L
                v_pos.append(pl.item())
                v_edge.append(el.item())
                v_succ.append(float(err.abs().max().item() < 0.05))

        vp = np.mean(v_pos); ve = np.mean(v_edge); vs = np.mean(v_succ)
        lr_now = sched.get_last_lr()[0]
        msg = (f"[{stage_name}] epoch {epoch:4d}/{args.epochs}  "
               f"train={t_loss/n:.5f}  val_pos={vp:.5f}  "
               f"val_edge={ve:.5f}  cold_succ@0.05={vs:.2f}  lr={lr_now:.2e}")
        print(msg, flush=True)
        with open(log_path, 'a') as f:
            f.write(json.dumps({'epoch':epoch,'train_loss':t_loss/n,
                'val_pos':vp,'val_edge':ve,'cold_succ':vs,'stage':stage_name})+'\n')

        if vp < best_val:
            best_val = vp
            ck_path = os.path.join(args.ckpt_dir, f'{stage_name}_best.pt')
            torch.save({'epoch':epoch,'model':model.state_dict(),
                'val_pos':vp,'cold_succ':vs,
                'h_dim':args.h_dim,'n_layers':args.n_layers}, ck_path)
            print(f"  ✓ checkpoint saved (val_pos={vp:.5f}, cold_succ={vs:.2f})")

    return os.path.join(args.ckpt_dir, f'{stage_name}_best.pt')


# ── Curriculum helper ─────────────────────────────────────────────────────────
def filter_by_ring(files, max_ring):
    """Keep only instances whose ring depth ≤ max_ring (parsed from filename)."""
    import re
    out = []
    for f in files:
        m = re.search(r'_r(\d+)_', os.path.basename(f))
        if m and int(m.group(1)) <= max_ring:
            out.append(f)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lib',  default='udg_library')
    ap.add_argument('--rings', type=int, nargs='+', default=[4,5])
    ap.add_argument('--curriculum', action='store_true',
                    help='train ring 1 → 2 → ... → max(rings) sequentially')
    ap.add_argument('--epochs',          type=int,   default=300)
    ap.add_argument('--epochs-per-stage',type=int,   default=150,
                    help='epochs per curriculum stage (overrides --epochs)')
    ap.add_argument('--h-dim',   type=int,   default=64)
    ap.add_argument('--n-layers',type=int,   default=6)
    ap.add_argument('--lr',      type=float, default=3e-4)
    ap.add_argument('--sigma-min', type=float, default=0.02)
    ap.add_argument('--sigma-max', type=float, default=5.25)
    ap.add_argument('--pos-weight',  type=float, default=1.0,
                    help='weight of Procrustes position loss (direct library supervision)')
    ap.add_argument('--edge-weight', type=float, default=0.5,
                    help='weight of edge-length loss (constraint satisfaction)')
    ap.add_argument('--T-eval',  type=int, default=4,
                    help='iterative refinement passes at eval time (train uses T=1)')
    ap.add_argument('--eval-every', type=int, default=10)
    ap.add_argument('--device', default='cuda' if
                    __import__('torch').cuda.is_available() else 'cpu')
    ap.add_argument('--seed',    type=int, default=0)
    ap.add_argument('--ckpt-dir', default='ckpt_denoise')
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--ckpt',    default='')
    args = ap.parse_args()

    print(f"Device: {args.device}")
    all_files = sorted(glob.glob(os.path.join(args.lib, '*.npz')))
    all_files = [f for f in all_files if not f.endswith('_base.npz')]
    print(f"Library: {len(all_files)} instances")

    model = DenoisingEmbedder(h_dim=args.h_dim, n_layers=args.n_layers,
                               sigma_min=args.sigma_min, sigma_max=args.sigma_max)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,d} parameters")

    if args.eval_only:
        # Cold-start evaluation with iterative refinement
        if not args.ckpt:
            sys.exit('--ckpt required for --eval-only')
        dev = torch.device(args.device)
        ck = torch.load(args.ckpt, map_location=dev)
        model.load_state_dict(ck['model']); model = model.to(dev); model.eval()
        files = filter_by_ring(all_files, max(args.rings))
        succ = []
        with torch.no_grad():
            for fname in files:
                d = np.load(fname)
                x0   = torch.tensor(d['positions'], dtype=torch.float32).to(dev)
                edges= torch.tensor(d['edges'],     dtype=torch.long).to(dev)
                L    = torch.tensor(d['lengths'],   dtype=torch.float32).to(dev)
                src, dst = edges[:,0], edges[:,1]
                x_cold = torch.randn_like(x0) * args.sigma_max
                pred = model(x_cold, src, dst, L, args.sigma_max, T_refine=args.T_eval)
                mx = (pred[src]-pred[dst]).norm(dim=-1).sub(L).abs().max().item()
                succ.append(mx < 0.05)
                print(f"{os.path.basename(fname)}: max_err={mx:.4f} {'✓' if mx<0.05 else '✗'}")
        print(f"\nCold-start succ@0.05: {np.mean(succ):.2f} ({sum(succ)}/{len(succ)})")
        return

    if args.curriculum:
        max_ring = max(args.rings)
        prev_ckpt = None
        orig_epochs = args.epochs
        args.epochs = args.epochs_per_stage
        for r in range(1, max_ring+1):
            stage_files = filter_by_ring(all_files, r)
            if not stage_files:
                print(f"No files for ring ≤ {r}, skipping")
                continue
            print(f"\n{'='*60}\nCurriculum stage: ring ≤ {r} ({len(stage_files)} files)\n{'='*60}")
            prev_ckpt = train_stage(model, stage_files, args,
                                    stage_name=f'ring{r}',
                                    prev_ckpt=prev_ckpt)
        args.epochs = orig_epochs
    else:
        files = filter_by_ring(all_files, max(args.rings))
        if not files:
            sys.exit(f"No files found for rings {args.rings} in {args.lib}")
        train_stage(model, files, args, stage_name='all')


if __name__ == '__main__':
    main()
