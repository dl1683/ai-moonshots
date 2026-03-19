"""Probe F: Stigmergic Text Modeling.

REVOLUTIONARY TEST: Can a system of locally-interacting compression agents
(no global attention, no central controller) learn to model text?

Inspired by: ant colonies, slime molds, mycorrhizal networks, immune systems.
All achieve intelligence through LOCAL interactions + shared medium.

Design:
- N=16 small "agents" each with ~600K params (total ~10M)
- Each agent sees a 32-token window
- Agents write compressed features to a shared 1D "medium"
- Agents read from their local neighborhood on the medium
- Multiple message-passing rounds (like pheromone accumulation)
- Compare against a single 10M-param transformer on same data

Controls:
- Random medium (agents write but read random positions)
- Single-agent baseline (one big model, no interaction)

Kill: If stigmergic model perplexity is >2x transformer baseline,
local interaction is insufficient for language modeling.
"""

import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
LEDGER_PATH = REPO_ROOT / "experiments" / "ledger.jsonl"


# =============================================================================
# Shared Data (same as Probe A for comparability)
# =============================================================================

class CharDataset(Dataset):
    def __init__(self, text, seq_len=256):
        self.data = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(1, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y


def get_text_data():
    random.seed(42)
    words = ["the", "cat", "sat", "on", "mat", "dog", "ran", "to", "big", "small",
             "red", "blue", "house", "tree", "car", "bird", "fish", "sun", "moon", "star",
             "if", "then", "because", "when", "while", "and", "or", "not", "but", "so",
             "every", "some", "no", "all", "each", "many", "few", "most", "any", "other",
             "is", "was", "has", "had", "will", "can", "may", "should", "could", "would",
             "think", "know", "see", "find", "give", "take", "make", "come", "go", "say"]
    sentences = []
    for _ in range(5000):
        length = random.randint(5, 20)
        sent = " ".join(random.choices(words, k=length))
        sentences.append(sent + ".")
    text = " ".join(sentences)
    split = int(len(text) * 0.9)
    return text[:split], text[split:]


# =============================================================================
# Standard Transformer Baseline (same as Probe A)
# =============================================================================

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        B, T, D = x.shape
        mask = torch.nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + h
        x = x + self.ff(self.ln2(x))
        return x


class TransformerBaseline(nn.Module):
    def __init__(self, vocab_size=256, dim=256, n_layers=6, n_heads=4, max_seq=256):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.max_seq = max_seq

    def forward(self, x):
        B, T = x.shape
        h = self.tok_emb(x) + self.pos_emb(torch.arange(T, device=x.device))
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# STIGMERGIC MODEL — The Revolutionary Architecture
# =============================================================================

class CompressionAgent(nn.Module):
    """A single local compression agent.

    Sees a small window of tokens, compresses them into a feature vector,
    reads from the local neighborhood of the shared medium, and writes
    an updated feature back.

    Think of it like an ant: it perceives locally, deposits pheromone,
    and reads nearby pheromone to adjust behavior.
    """

    def __init__(self, vocab_size, window_size, dim, medium_dim, n_layers=2):
        super().__init__()
        self.window_size = window_size
        self.dim = dim

        # Local encoder: compress window into feature
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(window_size, dim)
        self.encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # Medium reader: attend to local neighborhood features
        self.medium_proj = nn.Linear(medium_dim, dim)
        self.gate = nn.Linear(dim * 2, dim)  # Combine local + medium info

        # Medium writer: produce feature to deposit on medium
        self.writer = nn.Linear(dim, medium_dim)

        # Decoder: predict next tokens from combined representation
        self.decoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, vocab_size),
        )

    def encode_window(self, tokens):
        """Compress a window of tokens into a feature vector."""
        # tokens: (B, W)
        B, W = tokens.shape
        emb = self.tok_emb(tokens) + self.pos_emb(torch.arange(W, device=tokens.device))
        # Simple: mean pool after encoding
        h = self.encoder(emb)  # (B, W, dim)
        return h  # Return per-position features, not just pooled

    def read_medium(self, medium_features):
        """Read from the local neighborhood of the shared medium."""
        # medium_features: (B, neighborhood_size, medium_dim)
        h = self.medium_proj(medium_features)  # (B, N, dim)
        # Mean pool the neighborhood
        return h.mean(dim=1)  # (B, dim)

    def forward(self, tokens, medium_features):
        """Process one window: encode locally, integrate medium info, predict, write.

        Args:
            tokens: (B, W) - local window of token indices
            medium_features: (B, neighborhood_size, medium_dim) - nearby medium state

        Returns:
            logits: (B, W, vocab_size) - next-token predictions for this window
            deposit: (B, medium_dim) - feature to write to the medium
        """
        B, W = tokens.shape

        # 1. Compress local window
        local_features = self.encode_window(tokens)  # (B, W, dim)

        # 2. Read from medium
        medium_info = self.read_medium(medium_features)  # (B, dim)

        # 3. Gate: combine local and medium info
        # Broadcast medium_info across positions
        medium_expanded = medium_info.unsqueeze(1).expand(-1, W, -1)  # (B, W, dim)
        combined = torch.cat([local_features, medium_expanded], dim=-1)  # (B, W, dim*2)
        gated = torch.sigmoid(self.gate(combined)) * local_features + \
                (1 - torch.sigmoid(self.gate(combined))) * medium_expanded  # (B, W, dim)

        # 4. Predict next tokens
        logits = self.decoder(gated)  # (B, W, vocab_size)

        # 5. Write to medium (deposit pheromone)
        deposit = self.writer(local_features.mean(dim=1))  # (B, medium_dim)

        return logits, deposit


class StigmergicModel(nn.Module):
    """A colony of compression agents communicating through a shared medium.

    No global attention. No central controller. Intelligence emerges from
    local interactions through a shared representation medium.

    Architecture:
    - Input sequence is divided into windows
    - Each window is processed by a compression agent
    - Agents write to and read from a shared 1D medium
    - Multiple rounds of message passing allow information to propagate
    - Final predictions come from each agent's local output
    """

    def __init__(self, vocab_size=256, n_agents=16, window_size=16,
                 agent_dim=64, medium_dim=128, n_rounds=3, max_seq=256):
        super().__init__()
        self.n_agents = n_agents
        self.window_size = window_size
        self.medium_dim = medium_dim
        self.n_rounds = n_rounds
        self.max_seq = max_seq

        # All agents share weights (like how all ants follow the same rules)
        self.agent = CompressionAgent(vocab_size, window_size, agent_dim, medium_dim)

        # Medium: initialized to zeros, updated by agent deposits
        # The medium has one slot per window position
        self.medium_init = nn.Parameter(torch.zeros(1, n_agents, medium_dim))

        # Neighborhood size for medium reading (how far each agent can "smell")
        self.neighborhood = 3  # Read from 3 neighbors on each side

    def forward(self, x):
        """Process a full sequence through the stigmergic colony.

        Args:
            x: (B, T) - token indices

        Returns:
            logits: (B, T, vocab_size) - predictions
        """
        B, T = x.shape

        # Divide sequence into windows
        n_windows = T // self.window_size
        if n_windows == 0:
            n_windows = 1

        # Pad if needed
        padded_T = n_windows * self.window_size
        if padded_T < T:
            n_windows += 1
            padded_T = n_windows * self.window_size

        x_padded = F.pad(x, (0, padded_T - T), value=0)
        windows = x_padded.view(B, n_windows, self.window_size)  # (B, N_w, W)

        # Initialize medium
        medium = self.medium_init.expand(B, -1, -1)  # (B, n_agents, medium_dim)
        if n_windows > self.n_agents:
            medium = F.pad(medium, (0, 0, 0, n_windows - self.n_agents))
        elif n_windows < self.n_agents:
            medium = medium[:, :n_windows, :]

        # Message passing rounds (pheromone accumulation)
        all_logits = None
        for round_idx in range(self.n_rounds):
            round_logits = []
            new_medium = torch.zeros_like(medium)

            for w in range(n_windows):
                # Get this window's tokens
                window_tokens = windows[:, w, :]  # (B, W)

                # Get neighborhood from medium (CAUSAL: only look at previous windows)
                start = max(0, w - self.neighborhood)
                end = w  # Don't look at future windows (causal)
                if start == end:
                    # First window: use medium init
                    neighborhood = medium[:, w:w+1, :]  # (B, 1, medium_dim)
                else:
                    neighborhood = medium[:, start:end, :]  # (B, N, medium_dim)

                # Agent processes this window
                logits, deposit = self.agent(window_tokens, neighborhood)
                round_logits.append(logits)

                # Deposit on medium (accumulate, like pheromone)
                new_medium[:, w, :] = medium[:, w, :] + deposit

            # Update medium for next round
            medium = new_medium

            # Accumulate logits (last round's logits are the final output)
            all_logits = torch.cat(round_logits, dim=1)  # (B, N_w*W, vocab_size)

        # Trim to original sequence length
        return all_logits[:, :T, :]

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Training and Evaluation
# =============================================================================

def train_model(model, train_loader, epochs=15, lr=3e-4, name="model"):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    [{name}] Epoch {epoch+1}/{epochs}: loss={total_loss/n:.4f}", flush=True)
    return model


@torch.no_grad()
def measure_perplexity(model, test_loader):
    model.eval()
    total_loss = 0
    total_tokens = 0
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()
    return math.exp(total_loss / total_tokens)


@torch.no_grad()
def measure_bits_per_byte(model, test_loader):
    model.eval()
    total_loss = 0
    total_chars = 0
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_chars += y.numel()
    return total_loss / (total_chars * math.log(2))


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"Probe F: Stigmergic Text Modeling")
    print(f"Device: {DEVICE}")
    print(f"Can locally-interacting agents model text without global attention?")
    print(f"=" * 60)

    # Data
    train_text, test_text = get_text_data()
    train_ds = CharDataset(train_text, seq_len=256)
    test_ds = CharDataset(test_text, seq_len=256)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    results = []

    # --- Model 1: Transformer Baseline ---
    print(f"\n{'='*40}")
    print(f"Model 1: Transformer Baseline")
    print(f"{'='*40}")
    transformer = TransformerBaseline(vocab_size=256, dim=256, n_layers=6, n_heads=4).to(DEVICE)
    print(f"  Params: {transformer.count_params():,}")
    start = time.time()
    train_model(transformer, train_loader, epochs=15, name="Transformer")
    t_time = time.time() - start
    t_ppl = measure_perplexity(transformer, test_loader)
    t_bpb = measure_bits_per_byte(transformer, test_loader)
    print(f"  Perplexity: {t_ppl:.2f}")
    print(f"  Bits/byte: {t_bpb:.4f}")
    results.append({"model": "transformer_baseline", "params": transformer.count_params(),
                     "perplexity": round(t_ppl, 2), "bits_per_byte": round(t_bpb, 4),
                     "train_time_s": round(t_time, 1)})
    del transformer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Model 2: Stigmergic Colony ---
    print(f"\n{'='*40}")
    print(f"Model 2: Stigmergic Colony (16 agents, 3 rounds)")
    print(f"{'='*40}")
    stigmergic = StigmergicModel(
        vocab_size=256, n_agents=16, window_size=16,
        agent_dim=128, medium_dim=128, n_rounds=3, max_seq=256
    ).to(DEVICE)
    print(f"  Params: {stigmergic.count_params():,}")
    start = time.time()
    train_model(stigmergic, train_loader, epochs=15, name="Stigmergic")
    s_time = time.time() - start
    s_ppl = measure_perplexity(stigmergic, test_loader)
    s_bpb = measure_bits_per_byte(stigmergic, test_loader)
    print(f"  Perplexity: {s_ppl:.2f}")
    print(f"  Bits/byte: {s_bpb:.4f}")
    results.append({"model": "stigmergic_colony", "params": stigmergic.count_params(),
                     "perplexity": round(s_ppl, 2), "bits_per_byte": round(s_bpb, 4),
                     "train_time_s": round(s_time, 1)})
    del stigmergic
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Model 3: Stigmergic with Random Medium (Control) ---
    print(f"\n{'='*40}")
    print(f"Model 3: Stigmergic + Random Medium (Control)")
    print(f"{'='*40}")
    # Same architecture but medium features are shuffled randomly
    # This tests whether the MEDIUM COMMUNICATION matters or just having local agents is enough
    class RandomMediumModel(StigmergicModel):
        def forward(self, x):
            B, T = x.shape
            n_windows = max(1, T // self.window_size)
            padded_T = n_windows * self.window_size
            if padded_T < T:
                n_windows += 1
                padded_T = n_windows * self.window_size
            x_padded = F.pad(x, (0, padded_T - T), value=0)
            windows = x_padded.view(B, n_windows, self.window_size)
            medium = self.medium_init.expand(B, -1, -1)
            if n_windows > self.n_agents:
                medium = F.pad(medium, (0, 0, 0, n_windows - self.n_agents))
            elif n_windows < self.n_agents:
                medium = medium[:, :n_windows, :]
            all_logits = None
            for round_idx in range(self.n_rounds):
                round_logits = []
                new_medium = torch.zeros_like(medium)
                for w in range(n_windows):
                    window_tokens = windows[:, w, :]
                    # RANDOM: read from random positions instead of local neighborhood
                    rand_idx = torch.randint(0, n_windows, (min(self.neighborhood, n_windows),))
                    neighborhood = medium[:, rand_idx, :]
                    logits, deposit = self.agent(window_tokens, neighborhood)
                    round_logits.append(logits)
                    new_medium[:, w, :] = medium[:, w, :] + deposit
                medium = new_medium
                all_logits = torch.cat(round_logits, dim=1)
            return all_logits[:, :T, :]

    random_med = RandomMediumModel(
        vocab_size=256, n_agents=16, window_size=16,
        agent_dim=128, medium_dim=128, n_rounds=3, max_seq=256
    ).to(DEVICE)
    print(f"  Params: {random_med.count_params():,}")
    start = time.time()
    train_model(random_med, train_loader, epochs=15, name="RandomMedium")
    r_time = time.time() - start
    r_ppl = measure_perplexity(random_med, test_loader)
    r_bpb = measure_bits_per_byte(random_med, test_loader)
    print(f"  Perplexity: {r_ppl:.2f}")
    print(f"  Bits/byte: {r_bpb:.4f}")
    results.append({"model": "random_medium_control", "params": random_med.count_params(),
                     "perplexity": round(r_ppl, 2), "bits_per_byte": round(r_bpb, 4),
                     "train_time_s": round(r_time, 1)})
    del random_med
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"PROBE F RESULTS")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['model']:25s}: ppl={r['perplexity']:8.2f}  bpb={r['bits_per_byte']:.4f}  "
              f"params={r['params']:,}  time={r['train_time_s']}s")

    t_ppl_val = results[0]["perplexity"]
    s_ppl_val = results[1]["perplexity"]
    ratio = s_ppl_val / t_ppl_val
    print(f"\n  Stigmergic/Transformer perplexity ratio: {ratio:.2f}x")
    print(f"  Kill criterion (>2x): {'KILLED' if ratio > 2.0 else 'SURVIVED'}")

    if len(results) > 2:
        r_ppl_val = results[2]["perplexity"]
        print(f"  Random medium/Stigmergic ratio: {r_ppl_val/s_ppl_val:.2f}x")
        if r_ppl_val > s_ppl_val * 1.05:
            print(f"  MEDIUM COMMUNICATION MATTERS: random medium is {r_ppl_val/s_ppl_val:.2f}x worse")
        else:
            print(f"  Medium communication does NOT help: random medium is similar")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "probe": "F_stigmergic_text_modeling",
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "Locally-interacting compression agents can model text without global attention",
        "ppl_ratio": round(ratio, 4),
        "kill_criterion": "ratio > 2.0",
        "verdict": "KILLED" if ratio > 2.0 else "PROMISING" if ratio < 1.5 else "SURVIVED",
        "results": results,
    }
    out_path = RESULTS_DIR / "probe_f_stigmergic.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    ledger_entry = {
        "timestamp": datetime.now().isoformat(),
        "id": "probe_f_stigmergic",
        "purpose": "Can locally-interacting agents model text without global attention?",
        "command": "python code/probe_f_stigmergic.py",
        "metrics": {"ppl_ratio": round(ratio, 4), "transformer_ppl": t_ppl_val, "stigmergic_ppl": s_ppl_val},
        "status": "DONE",
        "notes": output["verdict"],
    }
    with open(LEDGER_PATH, "a") as f:
        f.write(json.dumps(ledger_entry) + "\n")

    print(f"\n  Results: {out_path}")


if __name__ == "__main__":
    main()
