"""Probe: Compositional Primitive Discovery.

Can we learn a small library of primitives that compose to cover a large space?
Like V(D)J covers the antigen space with 300 gene segments -> 10^11 antibodies.

Architecture:
1. LIBRARY of K primitive operations (small neural modules)
2. SELECTOR that picks which primitives to compose for each input
3. COMPOSER that chains the selected primitives

Training encourages:
- Primitives to be DIFFERENT (diversity loss)
- Primitives to be REUSABLE (each used for many inputs)
- Compositions to be MINIMAL (compression = use fewest primitives)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import json
import time
from pathlib import Path

torch.manual_seed(42)
random.seed(42)

DIM = 64
N_PRIMITIVES = 16
COMPOSE_DEPTH = 3
VOCAB = 64
SEQ = 32
REPO = Path(__file__).parent.parent


class PrimitiveLibrary(nn.Module):
    def __init__(self, dim, n_primitives):
        super().__init__()
        self.primitives = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
            for _ in range(n_primitives)
        ])
        self.n = n_primitives

    def apply_soft(self, x, weights):
        outputs = torch.stack([p(x) for p in self.primitives], dim=-1)
        return (outputs * weights.unsqueeze(1)).sum(-1)


class ComposerModel(nn.Module):
    def __init__(self, vocab_size, dim, n_prims, depth, max_seq=SEQ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(max_seq, dim)
        self.library = PrimitiveLibrary(dim, n_prims)
        self.selectors = nn.ModuleList([nn.Linear(dim, n_prims) for _ in range(depth)])
        self.depth = depth
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.n_prims = n_prims

    def forward(self, x, temperature=1.0):
        B, T = x.shape
        h = self.emb(x) + self.pos(torch.arange(T, device=x.device))
        diversity_loss = 0.0

        for d in range(self.depth):
            ctx = h.mean(dim=1)
            logits = self.selectors[d](ctx)
            weights = F.gumbel_softmax(logits, tau=temperature, hard=False)

            h_new = torch.zeros_like(h)
            for t in range(T):
                h_new[:, t, :] = self.library.apply_soft(h[:, t, :], weights)
            h = h + h_new

            avg_usage = weights.mean(0)
            uniform = torch.ones(self.n_prims, device=x.device) / self.n_prims
            diversity_loss += F.kl_div(avg_usage.log(), uniform, reduction="sum")

        return self.head(self.ln(h)), diversity_loss


class FlatModel(nn.Module):
    def __init__(self, vocab_size, dim, depth, max_seq=SEQ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(max_seq, dim)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
            for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x, temperature=None):
        B, T = x.shape
        h = self.emb(x) + self.pos(torch.arange(T, device=x.device))
        for layer in self.layers:
            h = h + layer(h)
        return self.head(self.ln(h)), 0.0


def make_data(n):
    data = []
    for _ in range(n):
        val = random.randint(1, 20)
        ops = random.choices(["add", "sub", "dbl", "hlf"], k=3)
        seq_chars = str(val)
        for op in ops:
            if op == "add":
                d = random.randint(1, 5)
                val += d
                seq_chars += "+" + str(val)
            elif op == "sub":
                d = random.randint(1, 3)
                val -= d
                seq_chars += "-" + str(val)
            elif op == "dbl":
                val *= 2
                seq_chars += "*" + str(val)
            elif op == "hlf":
                val = val // 2
                seq_chars += "/" + str(val)
        tokens = [ord(c) % VOCAB for c in seq_chars]
        tokens = (tokens + [0] * SEQ)[:SEQ]
        data.append(tokens)
    return torch.tensor(data)


def main():
    print("COMPOSITIONAL PRIMITIVE DISCOVERY")
    print("=" * 60)
    print(f"Library: {N_PRIMITIVES} primitives, composed {COMPOSE_DEPTH} deep")
    print()

    train = make_data(3000)
    test = make_data(500)

    results = []
    for name, model in [
        ("Composer", ComposerModel(VOCAB, DIM, N_PRIMITIVES, COMPOSE_DEPTH)),
        ("Flat", FlatModel(VOCAB, DIM, COMPOSE_DEPTH)),
    ]:
        params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(150):
            idx = torch.randperm(len(train))[:128]
            x = train[idx, :-1]
            y = train[idx, 1:]
            temp = max(0.5, 2.0 - epoch * 0.01)
            logits, aux_loss = model(x, temperature=temp)
            ce = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
            loss = ce + 0.01 * aux_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            if (epoch + 1) % 50 == 0:
                with torch.no_grad():
                    tl, _ = model(test[:200, :-1])
                    tloss = F.cross_entropy(tl.reshape(-1, VOCAB), test[:200, 1:].reshape(-1))
                print(f"  [{name}] Epoch {epoch+1}: test_loss={tloss.item():.4f}")

        with torch.no_grad():
            logits, _ = model(test[:200, :-1])
            final = F.cross_entropy(logits.reshape(-1, VOCAB), test[:200, 1:].reshape(-1))

        r = {"model": name, "params": params, "test_loss": round(final.item(), 4)}

        if hasattr(model, "selectors"):
            with torch.no_grad():
                h = model.emb(test[:50, :-1]) + model.pos(torch.arange(SEQ - 1))
                for d in range(COMPOSE_DEPTH):
                    sel = model.selectors[d](h.mean(1))
                    w = F.softmax(sel, dim=-1)
                    usage = w.mean(0)
                    active = (usage > 0.1).sum().item()
                    top3 = usage.topk(3)
                    print(f"  Depth {d}: active={active}/{N_PRIMITIVES}, "
                          f"top3=[{', '.join(f'P{i}:{v:.2f}' for i, v in zip(top3.indices.tolist(), top3.values.tolist()))}]")
                    r[f"depth_{d}_active"] = active

        print(f"{name:10s}: {params:>8,} params, final_loss={final.item():.4f}")
        results.append(r)
        print()

    # Save
    out = REPO / "results" / "probe_primitive_discovery.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump({"probe": "primitive_discovery", "results": results}, f, indent=2)
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
