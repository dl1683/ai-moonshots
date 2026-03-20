"""Tesla CPU Experiment: Multi-Teacher Stage-Mapped Absorption.
DeBERTa (encoder) -> Stage 5, GPT-2 (AR) -> Stage 7.
Compares CE-only vs dual-teacher on BPT + stage metrics.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, time, json, sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from sutra_v05_ssm import SutraV05, N_STAGES, top2_project

SEQ, BATCH, VOCAB, DIM, N_SEQ, STEPS = 64, 8, 50257, 128, 512, 200
ANCHOR_POS = [16, 48]
REPO = __import__('pathlib').Path(__file__).parent.parent

print("=== TESLA EXPERIMENT: Multi-Teacher Absorption ===\n")
tokens = torch.load(REPO / "data" / "minipile_tokens.pt", weights_only=True)

# Precompute teacher targets
print("Precomputing teacher targets...", flush=True)
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

enc_model = AutoModel.from_pretrained("microsoft/deberta-v3-base").eval()
enc_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
ar_model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
ar_tokenizer = AutoTokenizer.from_pretrained("gpt2")

enc_targets, ar_targets, data_x, data_y = [], [], [], []
for i in range(N_SEQ):
    s = i * (SEQ + 1)
    x, y = tokens[s:s+SEQ], tokens[s+1:s+SEQ+1]
    data_x.append(x); data_y.append(y)
    text = ar_tokenizer.decode(x.tolist())
    enc_in = enc_tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        eh = enc_model(**enc_in).last_hidden_state
    el = eh.size(1)
    enc_targets.append(eh[0, [min(16,el-1), min(48,el-1)], :].clone())
    with torch.no_grad():
        ar_targets.append(ar_model(x.unsqueeze(0)).logits[0].clone())
    if (i+1) % 128 == 0: print(f"  {i+1}/{N_SEQ}", flush=True)

data_x, data_y = torch.stack(data_x), torch.stack(data_y)
del enc_model, ar_model
print(f"Done: {len(enc_targets)} sequences\n")

class Stage5Proj(nn.Module):
    def __init__(self, d=128, td=768):
        super().__init__()
        self.proj = nn.Linear(d, td)
    def forward(self, x): return self.proj(x)

results = {}
for mode in ["CE_only", "Dual_teacher"]:
    torch.manual_seed(42)
    model = SutraV05(vocab_size=VOCAB, dim=DIM, ff_dim=DIM*2, max_steps=3, window=2, k_retrieval=4)
    proj = Stage5Proj(DIM, 768) if mode == "Dual_teacher" else None
    params = list(model.parameters()) + (list(proj.parameters()) if proj else [])
    opt = torch.optim.AdamW(params, lr=8e-4, weight_decay=0.01)
    t0 = time.time()

    for step in range(STEPS):
        idx = torch.randint(0, N_SEQ, (BATCH,))
        x, y = data_x[idx], data_y[idx]
        logits, _ = model(x)
        Tc = min(logits.size(1), y.size(1))
        ce = F.cross_entropy(logits[:,:Tc].reshape(-1, VOCAB), y[:,:Tc].reshape(-1))
        loss = ce

        if mode == "Dual_teacher" and step >= STEPS * 0.3:
            h = model.ln(model.init_mu(model.emb(x) + model.pos_emb(torch.arange(SEQ))))
            enc_loss = torch.tensor(0.0)
            for b in range(BATCH):
                th = enc_targets[idx[b]]
                for ai, ap in enumerate(ANCHOR_POS):
                    if ap < h.size(1):
                        sp = proj(h[b, ap])
                        enc_loss = enc_loss + 1 - F.cosine_similarity(sp.unsqueeze(0), th[ai].unsqueeze(0))
            loss = loss + 0.05 * enc_loss / (BATCH * len(ANCHOR_POS))

        if mode == "Dual_teacher" and step >= STEPS * 0.55:
            tl = torch.stack([ar_targets[idx[b]] for b in range(BATCH)])
            T = 2.0
            ar_kd = F.kl_div(F.log_softmax(logits[:,:Tc]/T, dim=-1), F.softmax(tl[:,:Tc]/T, dim=-1), reduction="batchmean") * T*T
            loss = loss + 0.08 * ar_kd

        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad()
        if (step+1) % 50 == 0:
            print(f"  {mode} step {step+1}: BPT={ce.item()/math.log(2):.3f}", flush=True)

    model.eval()
    with torch.no_grad():
        tl, _ = model(data_x[:64])
        test_bpt = F.cross_entropy(tl.reshape(-1, VOCAB), data_y[:64,:tl.size(1)].reshape(-1)).item() / math.log(2)
        h = model.emb(data_x[:4]) + model.pos_emb(torch.arange(SEQ))
        mu = model.init_mu(h); lam = F.softplus(model.init_lam(h))+0.1
        pi = torch.zeros(4,SEQ,7); pi[:,:,2]=1.0
        for t in range(3):
            K = model.transition(mu)
            pi_ev = torch.bmm(pi.view(4*SEQ,1,7), K.view(4*SEQ,7,7)).view(4,SEQ,7)
            so, ev = model.stage_bank(mu, pi)
            pi_new = pi_ev * F.softmax(ev/0.5, dim=-1)
            pi = top2_project(pi_new / pi_new.sum(-1,keepdim=True).clamp(min=1e-8))
            msgs = model.router(mu) * pi[:,:,3:4]
            mu, lam = model.writer(mu, lam, msgs, pi[:,:,4:5])
            mu = mu + so * 0.1
        se = -(pi*((pi+1e-10).log())).sum(-1).mean().item()

    results[mode] = {"bpt": round(test_bpt,3), "stage_entropy": round(se,4), "time": round(time.time()-t0,1)}
    print(f"{mode}: BPT={test_bpt:.3f}, stage_ent={se:.4f}\n")

print("=== RESULTS ===")
for m, r in results.items(): print(f"  {m:>15s}: BPT={r['bpt']:.3f}, stage_ent={r['stage_entropy']:.4f}")
adv = (results["CE_only"]["bpt"] - results["Dual_teacher"]["bpt"]) / results["CE_only"]["bpt"] * 100
ec = (results["Dual_teacher"]["stage_entropy"] - results["CE_only"]["stage_entropy"]) / max(results["CE_only"]["stage_entropy"], 0.001) * 100
print(f"\nDual-teacher BPT advantage: {adv:+.1f}%")
print(f"Stage entropy change: {ec:+.1f}%")
if adv > 0 and ec > -10: print("VERDICT: ABSORPTION WORKS")
elif adv > 0: print("VERDICT: TRAP (stages collapsed)")
else: print("VERDICT: NO BENEFIT")
json.dump(results, open(REPO / "results" / "tesla_experiment.json", "w"), indent=2)
