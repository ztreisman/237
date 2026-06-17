#!/usr/bin/env python3
"""
train_udg_embedder.py — Train an EGNN to embed unit-distance graphs in R^d.

Self-contained: generates its own training library if none is provided,
then trains an E(n)-equivariant GNN to map abstract graphs to unit-edge
embeddings in R^d with d as small as possible.

Works locally (CPU) and on Google Colab (GPU). On Colab:
  1. Upload this file and generate_unit_distance_library.py
  2. !python train_udg_embedder.py --epochs 200 --device cuda

ARCHITECTURE — Equivariant GNN (EGNN, Satorras et al. 2021):
  Input:  graph topology, per-edge target lengths, random initial positions
  Layers: T rounds of (edge messages → coordinate update → node feature update)
  Output: refined positions in R^d
  Loss:   Σ (||xi−xj|| − L_ij)² / |E|   (edge-length MSE)
  E(n)-equivariance: coordinate outputs transform correctly under rotations
  and translations by construction (no explicit symmetrization needed).

BASELINE: physics relaxer (L-BFGS on same loss from same random init).
SUCCESS METRIC: max|edge err| < 0.05 (stricter than the library's 0.02
certification — measures generalization quality).

TRAIN/VAL SPLIT: by base_id — the same embedded surface never appears on
both sides of the split, preventing the model from memorizing layouts.
"""

import argparse, json, math, os, sys, time, glob
import numpy as np

# ── optional torch import with helpful error ──────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    sys.exit("PyTorch not found. Install with: pip install torch")

# ── data generation (reuses library generator if available) ───────────────
def import_generator():
    for path in ['.', os.path.dirname(__file__)]:
        candidate = os.path.join(path, 'generate_unit_distance_library.py')
        if os.path.exists(candidate):
            import importlib.util
            spec = importlib.util.spec_from_file_location('gen', candidate)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    return None


def generate_library(out_dir, trials=20, rings=(4, 5), qs=(7,), dims=(3,)):
    gen = import_generator()
    if gen is None:
        sys.exit("generate_unit_distance_library.py not found in current "
                 "directory. Please place it alongside this script.")
    os.makedirs(out_dir, exist_ok=True)
    manifest = os.path.join(out_dir, 'manifest.jsonl')
    for q in qs:
        for dim in dims:
            for r in rings:
                adj, ring_of, ring_lists, ach = gen.build_3q_disk(
                    q, r, max_verts=10000)
                for trial in range(trials):
                    base_id = f"q{q}_n{dim}_r{ach}_t{trial:03d}"
                    base_npz = os.path.join(out_dir, base_id + '_base.npz')
                    if os.path.exists(base_npz):
                        data = np.load(base_npz)
                        P, edges = data['positions'], data['edges']
                    else:
                        P = gen.grow_embedding(adj, ring_lists, ach, dim,
                                               seed=trial)
                        edges = gen.edges_from_adj(adj)
                        L = np.ones(len(edges))
                        _, std, mx = gen.edge_stats(P, edges, L)
                        if std > 0.06:
                            continue
                        np.savez_compressed(base_npz, positions=P,
                                            edges=edges, lengths=L)
                        with open(manifest, 'a') as f:
                            f.write(json.dumps({'file': base_id+'_base.npz',
                                'base_id': base_id, 'kind': 'base',
                                'q': q, 'dim': dim}) + '\n')
                        print(f"  base {base_id}: V={len(P)} std={std:.4f}",
                              flush=True)
                    # chord augmentation
                    rng = np.random.default_rng(1000 + trial)
                    edges_set = {(int(a), int(b)) for a, b in edges}
                    for target_L in [1.0, 2.0]:
                        cands = gen.chord_candidates(
                            P, adj, edges_set, target_L, 0.03, 4, 2000, rng)
                        if not cands:
                            continue
                        for dens in [0.2, 0.6, 1.0]:
                            k = max(1, int(dens * len(cands)))
                            sel = rng.choice(len(cands), k, replace=False)
                            chords = np.array([cands[i] for i in sel],
                                              dtype=np.int64)
                            E2 = np.vstack([edges, chords])
                            L2 = np.concatenate([np.ones(len(edges)),
                                                 np.full(len(chords),
                                                         target_L)])
                            P2 = gen.relax(P.copy(), E2, L2, iters=400)
                            _, _, mx = gen.edge_stats(P2, E2, L2)
                            if mx > 0.02:
                                continue
                            tag = (f"{base_id}_L{target_L:g}"
                                   f"_d{int(100*dens):03d}")
                            npz = os.path.join(out_dir, tag + '.npz')
                            np.savez_compressed(npz, positions=P2, edges=E2,
                                                lengths=L2)
                            with open(manifest, 'a') as f:
                                f.write(json.dumps({
                                    'file': tag+'.npz', 'base_id': base_id,
                                    'kind': 'augmented', 'q': q, 'dim': dim,
                                    'n_chords': len(chords),
                                    'chord_L': target_L}) + '\n')
                            print(f"  aug {tag}: +{len(chords)} chords "
                                  f"max={mx:.4f}", flush=True)
    return manifest


# ── dataset ───────────────────────────────────────────────────────────────
class UDGDataset(torch.utils.data.Dataset):
    def __init__(self, npz_files, val=False, val_frac=0.15, seed=0):
        # split by base_id prefix to prevent surface leakage
        # base_id = q{q}_n{dim}_r{ring}_t{trial} (first 4 tokens) -- using
        # 5 tokens would also include the chord-length tag (L1/L2) and
        # split variants of the SAME surface across train/val.
        by_base = {}
        for f in npz_files:
            base = '_'.join(os.path.basename(f).split('_')[:4])
            by_base.setdefault(base, []).append(f)
        bases = sorted(by_base)
        rng = np.random.default_rng(seed)
        rng.shuffle(bases)
        n_val = max(1, int(len(bases) * val_frac))
        val_bases = set(bases[:n_val])
        chosen = {b for b in bases if (b in val_bases) == val}
        self.files = [f for b in chosen for f in by_base[b]]
        print(f"{'Val' if val else 'Train'}: {len(self.files)} instances "
              f"from {len(chosen)} surfaces")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        data = np.load(self.files[i])
        pos = torch.tensor(data['positions'], dtype=torch.float32)
        edges = torch.tensor(data['edges'], dtype=torch.long)
        lengths = torch.tensor(data['lengths'], dtype=torch.float32)
        return pos, edges, lengths


def collate_fn(batch):
    """Variable-size graphs: return list (no padding needed for our model)."""
    return batch


# ── model: E(n)-equivariant GNN ───────────────────────────────────────────
class EGNNLayer(nn.Module):
    def __init__(self, h_dim, e_dim=2):
        super().__init__()
        # edge network: (h_i, h_j, ||xi-xj||^2, L_ij, L_ij^2) → message
        self.edge_net = nn.Sequential(
            nn.Linear(2 * h_dim + 3, h_dim), nn.SiLU(),
            nn.Linear(h_dim, h_dim), nn.SiLU())
        # coordinate update: message → scalar weight
        self.coord_net = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2), nn.SiLU(),
            nn.Linear(h_dim // 2, 1))
        # node update: (h_i, aggregated messages) → h_i'
        self.node_net = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim), nn.SiLU(),
            nn.Linear(h_dim, h_dim))
        # EGNN-style: start the coordinate-update output near zero so early
        # training doesn't move vertices by large, untrained-network amounts
        # (Satorras et al. 2021) -- without this the first few steps can
        # blow up positions before the network has learned anything useful.
        nn.init.xavier_uniform_(self.coord_net[-1].weight, gain=0.001)
        nn.init.zeros_(self.coord_net[-1].bias)

    def forward(self, h, x, edge_src, edge_dst, L):
        # x: (V, d), h: (V, h_dim), L: (E,)
        dv = x[edge_src] - x[edge_dst]           # (E, d)
        dist = torch.sqrt((dv * dv).sum(-1, keepdim=True) + 1e-12)  # (E, 1)
        dv_hat = dv / dist
        # relative distance error, bounded to (-1, 1) and zero at the
        # target: 0 when dist==L, saturating gracefully as dist->0 or
        # dist->inf. Raw d2 (O(scale^2), e.g. >100 for cold-init scale~4)
        # let x's scale feed back into d2 of the next layer -- an unbounded
        # exponential blow-up across layers. An unbounded ratio dist/L had
        # the same problem one step removed: after ~100-200 Adam steps it
        # drove coord_net's pre-tanh output into saturation for nearly all
        # edges at once, killing gradients everywhere before anything
        # useful was learned.
        rel_d = torch.tanh(dist / (L.unsqueeze(-1) + 1e-6) - 1.0)
        e_feat = torch.cat([h[edge_src], h[edge_dst], rel_d,
                             L.unsqueeze(-1),
                             (L * L).unsqueeze(-1)], dim=-1)
        msg = self.edge_net(e_feat)               # (E, h)
        # coordinate update: bounded step (|w|<1) of at most one target
        # edge-length, along the current src->dst direction. Bounding caps
        # each layer's position change regardless of how large msg/rel_d
        # get, breaking the blow-up feedback loop above. softsign (not
        # tanh): tanh's gradient decays exponentially (tanh'(5)~2e-4), so
        # the large initial loss (cold-init can give loss~30) drove
        # coord_net's output into tanh's saturated regime within ~150
        # steps and flatlined gradients network-wide. softsign's gradient
        # decays only as 1/(1+|x|)^2, staying useful over a much wider
        # range of pre-activation magnitudes. The /4 widens softsign's
        # near-linear region 4x (coord_net's raw output now needs |x|~4
        # before strongly saturating, not |x|~1), giving more headroom for
        # gradients to stay alive over a long (150-epoch) run.
        w = F.softsign(self.coord_net(msg) * 0.25)  # (E, 1)
        step = w * dv_hat * L.unsqueeze(-1)        # (E, d)
        delta_x = torch.zeros_like(x)
        delta_x.scatter_add_(0, edge_src.unsqueeze(-1).expand_as(dv), step)
        deg = torch.bincount(edge_src, minlength=x.shape[0]).float().clamp(min=1)
        x = x + delta_x / deg.unsqueeze(-1)
        # node update
        agg = torch.zeros_like(h)
        agg.scatter_add_(0, edge_src.unsqueeze(-1).expand_as(msg), msg)
        h = h + self.node_net(torch.cat([h, agg], dim=-1))
        return h, x


class UDGEmbedder(nn.Module):
    def __init__(self, h_dim=64, n_layers=6, max_dim=4):
        super().__init__()
        self.h_dim = h_dim
        self.max_dim = max_dim
        # node encoder: degree (log) + ring-distance proxy
        self.node_enc = nn.Linear(1, h_dim)
        self.layers = nn.ModuleList(
            [EGNNLayer(h_dim) for _ in range(n_layers)])

    def forward(self, x_init, edge_src, edge_dst, L):
        """
        x_init: (V, d) random initial positions
        edge_src, edge_dst: (E,) edge indices, one direction per undirected
                            edge (edge_src < edge_dst, as produced by
                            edges_from_adj)
        L: (E,) target lengths
        Returns: (V, d) refined positions
        """
        V = x_init.shape[0]
        # node features: log-degree (simple, graph-structural). With a
        # single direction per edge, degree = count as src + count as dst.
        deg = (torch.bincount(edge_src, minlength=V) +
               torch.bincount(edge_dst, minlength=V)).float()
        h = self.node_enc(torch.log1p(deg).unsqueeze(-1))
        x = x_init
        # EGNNLayer's coordinate/message update only scatters to edge_src,
        # so mirror each edge: with only (src,dst), the larger-indexed
        # endpoint of every edge would never receive a direct positional
        # update and stay frozen at its (random) init forever.
        src2 = torch.cat([edge_src, edge_dst])
        dst2 = torch.cat([edge_dst, edge_src])
        L2 = torch.cat([L, L])
        for layer in self.layers:
            h, x = layer(h, x, src2, dst2, L2)
        return x


# ── loss and evaluation ───────────────────────────────────────────────────
def edge_length_loss(pos, edge_src, edge_dst, L):
    # torch.norm's gradient at exactly 0 is NaN (0/0); with cold-init,
    # optimization can drive connected vertices arbitrarily close together,
    # so stabilize with a small epsilon under the sqrt.
    diff = pos[edge_src] - pos[edge_dst]
    d = torch.sqrt((diff * diff).sum(-1) + 1e-12)
    return ((d - L) ** 2).mean()


def edge_stats(pos, edge_src, edge_dst, L):
    with torch.no_grad():
        d = (pos[edge_src] - pos[edge_dst]).norm(dim=-1)
        err = (d - L).abs()
        return float(err.mean()), float(err.std()), float(err.max())


def relaxer_baseline(pos_np, edges_np, L_np, iters=300):
    """Run the physics relaxer on the same random init for comparison."""
    gen = import_generator()
    if gen is None:
        return None
    return gen.relax(pos_np.copy(), edges_np, L_np, iters=iters)


# ── training ─────────────────────────────────────────────────────────────
def train(args):
    device = torch.device(args.device)
    rng_torch = torch.Generator()
    rng_torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── data ──
    if args.lib:
        npz_files = []
        for lib in args.lib:
            files = sorted(glob.glob(os.path.join(lib, '*.npz')))
            npz_files += [f for f in files if not f.endswith('_base.npz')
                          or args.include_base]
    else:
        print("No library provided — generating one now...")
        manifest = generate_library(
            args.gen_out,
            trials=args.gen_trials,
            rings=tuple(args.gen_rings),
            qs=tuple(args.gen_q),
            dims=tuple(args.gen_dims))
        npz_files = sorted(glob.glob(os.path.join(args.gen_out, '*.npz')))
        npz_files = [f for f in npz_files if not f.endswith('_base.npz')]

    if not npz_files:
        sys.exit("No NPZ files found. Check --lib path or generation settings.")
    print(f"\nTotal instances: {len(npz_files)}")

    train_set = UDGDataset(npz_files, val=False, seed=args.seed)
    val_set = UDGDataset(npz_files, val=True, seed=args.seed)

    # infer ambient dimension from data
    sample = np.load(train_set.files[0])
    d = sample['positions'].shape[1]
    print(f"Ambient dimension: {d}\n")

    # ── model ──
    model = UDGEmbedder(h_dim=args.h_dim, n_layers=args.n_layers,
                        max_dim=d).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,d} parameters")

    # AdamW (decoupled weight decay) keeps coord_net's weights from
    # drifting unboundedly over a long run, which otherwise pushes
    # softsign's input further into saturation and flatlines gradients
    # (observed: val_mean frozen to 1e-5 precision from epoch 10-50).
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    # Warm restarts (not a single cosine decay to ~0): a plain
    # CosineAnnealingLR over `epochs` guarantees the LR is near-frozen by
    # the end, so a long run just converges precisely to whichever local
    # minimum it finds early and stays there (observed: val_mean frozen to
    # 1e-5 precision from epoch 10 on). Restarting the cosine cycle every
    # `lr_restart_period` epochs gives a long run repeated chances to
    # escape such a minimum.
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=args.lr_restart_period, eta_min=args.lr * 0.05)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_val = float('inf')
    np_rng = np.random.default_rng(args.seed)

    for epoch in range(1, args.epochs + 1):
        # ── train ──
        model.train()
        np_rng.shuffle(train_set.files)
        t_loss = 0.0; n = 0
        opt.zero_grad()
        in_batch = 0
        for fi, fname in enumerate(train_set.files):
            data = np.load(fname)
            pos_gt = torch.tensor(data['positions'], dtype=torch.float32)
            edges = torch.tensor(data['edges'], dtype=torch.long)
            L = torch.tensor(data['lengths'], dtype=torch.float32)
            V = pos_gt.shape[0]
            if args.cold_init:
                # purely random positions, independent of ground truth:
                # the model sees only combinatorial data (edges, lengths)
                scale = np_rng.uniform(*args.cold_scale)
                x_init = torch.tensor(np_rng.normal(0, scale, (V, d)),
                                      dtype=torch.float32).to(device)
            else:
                # random initial positions (recentered ground truth + large
                # noise encourages the model to learn from structure, not
                # memorize init)
                noise_scale = 0.5 + np_rng.random() * 1.5
                x_init = (pos_gt + torch.randn(V, d) * noise_scale).to(device)
                # center and scale
                x_init = x_init - x_init.mean(0)
            src, dst = edges[:, 0].to(device), edges[:, 1].to(device)
            L_dev = L.to(device)
            pred = model(x_init, src, dst, L_dev)
            loss = edge_length_loss(pred, src, dst, L_dev)
            # gradient accumulation over args.batch_size instances:
            # cold_scale in [1,4] makes per-instance loss magnitudes vary
            # ~70x, so per-instance (batch=1) gradient *directions* are
            # very inconsistent. Adam's momentum (m_hat) averages these
            # toward ~0 even when individual grad norms aren't tiny --
            # observed as train_loss still fluctuating while val_mean
            # stays frozen to 1e-5 precision for 40+ epochs. Averaging
            # gradients over a batch before each step gives Adam's
            # momentum a consistent signal to track.
            (loss / args.batch_size).backward()
            in_batch += 1
            if in_batch == args.batch_size or fi == len(train_set.files) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                opt.zero_grad()
                in_batch = 0
            t_loss += loss.item(); n += 1
        sched.step()

        if epoch % args.eval_every != 0:
            continue

        # ── validate ──
        model.eval()
        v_losses, v_max_errs, v_stds = [], [], []
        bl_max_errs = []
        with torch.no_grad():
            for fname in val_set.files[:min(30, len(val_set.files))]:
                data = np.load(fname)
                pos_gt = torch.tensor(data['positions'], dtype=torch.float32)
                edges_np = data['edges']
                L_np = data['lengths']
                edges = torch.tensor(edges_np, dtype=torch.long)
                L = torch.tensor(L_np, dtype=torch.float32)
                V = pos_gt.shape[0]
                # fixed eval init: same seed for fair comparison
                g = torch.Generator().manual_seed(42)
                if args.cold_init:
                    # purely random, independent of ground truth
                    scale = sum(args.cold_scale) / 2
                    x_init = torch.randn(V, d, generator=g) * scale
                else:
                    x_init = pos_gt + torch.randn(V, d, generator=g) * 1.0
                    x_init = x_init - x_init.mean(0)
                src = edges[:, 0]; dst = edges[:, 1]
                pred = model(x_init.to(device), src.to(device),
                             dst.to(device), L.to(device))
                mean_e, std_e, max_e = edge_stats(
                    pred.cpu(), src, dst, L)
                v_losses.append(mean_e); v_stds.append(std_e)
                v_max_errs.append(max_e)
                # relaxer baseline (same init, same budget)
                if args.eval_baseline:
                    P_bl = relaxer_baseline(
                        x_init.numpy(), edges_np, L_np,
                        iters=args.baseline_iters)
                    if P_bl is not None:
                        P_bl_t = torch.tensor(P_bl, dtype=torch.float32)
                        _, _, bl_mx = edge_stats(P_bl_t, src, dst, L)
                        bl_max_errs.append(bl_mx)

        v_mean = np.mean(v_losses)
        v_maxerr = np.mean(v_max_errs)
        v_succ = np.mean([e < 0.05 for e in v_max_errs])
        bl_str = ""
        if bl_max_errs:
            bl_succ = np.mean([e < 0.05 for e in bl_max_errs])
            bl_str = f"  baseline_succ={bl_succ:.2f}"
        print(f"epoch {epoch:4d}/{args.epochs}  "
              f"train_loss={t_loss/n:.5f}  "
              f"val_mean={v_mean:.5f}  val_max={v_maxerr:.4f}  "
              f"val_succ@0.05={v_succ:.2f}{bl_str}  "
              f"lr={sched.get_last_lr()[0]:.2e}", flush=True)

        if v_mean < best_val:
            best_val = v_mean
            ckpt = os.path.join(args.ckpt_dir, 'best.pt')
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'val_mean': v_mean, 'val_succ': v_succ,
                        'h_dim': args.h_dim, 'n_layers': args.n_layers,
                        'd': d}, ckpt)
            print(f"  ✓ saved best checkpoint (val_mean={v_mean:.5f})")

    print("\nTraining complete. Best checkpoint:", ckpt)


# ── main ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument('--lib', nargs='+', default=[],
                    help='path(s) to existing NPZ library dir(s)')
    ap.add_argument('--include-base', action='store_true',
                    help='include non-augmented base embeddings')
    # generation (if no --lib)
    ap.add_argument('--gen-out', default='udg_train_lib')
    ap.add_argument('--gen-trials', type=int, default=20)
    ap.add_argument('--gen-rings', type=int, nargs='+', default=[4, 5])
    ap.add_argument('--gen-q', type=int, nargs='+', default=[7])
    ap.add_argument('--gen-dims', type=int, nargs='+', default=[3])
    # model
    ap.add_argument('--h-dim', type=int, default=64)
    ap.add_argument('--n-layers', type=int, default=6)
    # training
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--lr-restart-period', type=int, default=30,
                    help='epochs per cosine warm-restart cycle')
    ap.add_argument('--batch-size', type=int, default=16,
                    help='instances per gradient step (accumulated)')
    ap.add_argument('--device', default='cuda' if
                    __import__('torch').cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ckpt-dir', default='checkpoints')
    ap.add_argument('--eval-every', type=int, default=10)
    ap.add_argument('--eval-baseline', action='store_true',
                    help='compare against physics relaxer at eval (slower)')
    ap.add_argument('--baseline-iters', type=int, default=300,
                    help='L-BFGS iterations for the relaxer baseline')
    ap.add_argument('--cold-init', action='store_true',
                    help='initialize positions randomly, independent of '
                         'ground truth (model sees combinatorial data only)')
    ap.add_argument('--cold-scale', type=float, nargs=2, default=[1.0, 4.0],
                    help='range for random init std in --cold-init mode')
    args = ap.parse_args()
    print(f"Device: {args.device}")
    train(args)


if __name__ == '__main__':
    main()
