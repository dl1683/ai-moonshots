"""Pre-built analysis scripts for probe results.

When probe JSONs appear, run this to get instant analysis.
No waiting to write analysis code AFTER results arrive.

Usage:
    python code/analysis_probes.py  # Analyze all available results
    python code/analysis_probes.py --probe a  # Analyze specific probe
"""

import json
import math
import sys
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results"


def analyze_probe_a():
    """Analyze compression ↔ capability correlation."""
    path = RESULTS / "probe_a_compression_capability.json"
    if not path.exists():
        print("Probe A: not yet complete")
        return None

    with open(path) as f:
        d = json.load(f)

    print("=" * 60)
    print("PROBE A: COMPRESSION ↔ CAPABILITY")
    print("=" * 60)
    print(f"Verdict: {d['verdict']}")
    print(f"r(compression, capability): {d['r_compression_capability']}")
    print()
    print(f"{'Condition':<20s} {'BPB':>8s} {'Reasoning Acc':>14s}")
    print("-" * 45)
    for c in d["conditions"]:
        print(f"{c['condition']:<20s} {c['bits_per_byte']:>8.4f} {c['reasoning_accuracy']:>14.4f}")

    r = d["r_compression_capability"]
    if r > 0.8:
        print(f"\nSTRONG: Compression strongly predicts capability (r={r:.3f})")
        print("Thesis CONFIRMED at this scale. MDL training direction is promising.")
    elif r > 0.5:
        print(f"\nMODERATE: Some relationship (r={r:.3f})")
        print("Thesis has directional support but not definitive.")
    elif r > 0.0:
        print(f"\nWEAK: Slight positive relationship (r={r:.3f})")
        print("Thesis not killed but not strongly supported either.")
    else:
        print(f"\nNEGATIVE/ZERO: No relationship (r={r:.3f})")
        print("Thesis KILLED at this scale. Compression ≠ capability here.")

    return d


def analyze_probe_f():
    """Analyze stigmergic text modeling."""
    path = RESULTS / "probe_f_stigmergic.json"
    if not path.exists():
        print("Probe F: not yet complete")
        return None

    with open(path) as f:
        d = json.load(f)

    print("=" * 60)
    print("PROBE F: STIGMERGIC TEXT MODELING")
    print("=" * 60)
    print(f"Verdict: {d['verdict']}")
    print(f"Perplexity ratio (stigmergic/transformer): {d['ppl_ratio']:.4f}x")
    print()
    for r in d["results"]:
        print(f"  {r['model']:<25s}: ppl={r['perplexity']:>8.2f}  bpb={r['bits_per_byte']:.4f}  "
              f"params={r['params']:,}")

    ratio = d["ppl_ratio"]
    if ratio < 1.0:
        print(f"\nSTIGMERGIC BEATS TRANSFORMER! Ratio={ratio:.3f}")
        print("Local-only processing is SUPERIOR. Revolutionary result.")
    elif ratio < 1.1:
        print(f"\nCLOSE MATCH: Ratio={ratio:.3f}")
        print("Stigmergic competitive with transformer. Core direction viable.")
    elif ratio < 1.5:
        print(f"\nWORSE BUT VIABLE: Ratio={ratio:.3f}")
        print("Gap exists but may close with better design. Direction worth pursuing.")
    elif ratio < 2.0:
        print(f"\nSIGNIFICANT GAP: Ratio={ratio:.3f}")
        print("Stigmergic substantially worse. Needs scratchpad or rethinking.")
    else:
        print(f"\nKILLED: Ratio={ratio:.3f}")
        print("Local-only is fundamentally insufficient. Need global mechanism.")

    # Check random medium control
    if len(d["results"]) > 2:
        stig_ppl = d["results"][1]["perplexity"]
        rand_ppl = d["results"][2]["perplexity"]
        if rand_ppl > stig_ppl * 1.05:
            print(f"\nMEDIUM MATTERS: random medium {rand_ppl/stig_ppl:.2f}x worse")
            print("Communication through the medium IS providing useful information.")
        else:
            print(f"\nMEDIUM DOESN'T HELP: random ≈ structured ({rand_ppl/stig_ppl:.2f}x)")
            print("Agents aren't using the medium effectively.")

    return d


def analyze_probe_e():
    """Analyze tokenization results."""
    path = RESULTS / "probe_e_tokenization.json"
    if not path.exists():
        print("Probe E: not yet complete")
        return None

    with open(path) as f:
        d = json.load(f)

    print("=" * 60)
    print("PROBE E: TOKENIZATION ANALYSIS")
    print("=" * 60)
    for r in d["results"]:
        print(f"  {r['method']:<25s}: BPB={r['bits_per_byte']:.4f}  "
              f"chars/tok={r['compression_ratio']:.2f}  vocab={r['vocab_size']}")

    return d


def analyze_mvp():
    """Analyze Sutra v0.2-MVP results."""
    path = RESULTS / "sutra_v02_mvp.json"
    if not path.exists():
        print("Sutra v0.2-MVP: not yet complete")
        return None

    with open(path) as f:
        d = json.load(f)

    print("=" * 60)
    print("SUTRA v0.2-MVP: THE CRITICAL TEST")
    print("=" * 60)
    for r in d["results"]:
        print(f"  {r['model']:<35s}: BPB={r['bpb']:.4f}  params={r['params']:,}")

    baseline = d["results"][0]["bpb"]
    for r in d["results"][1:]:
        ratio = r["bpb"] / baseline
        status = "BETTER!" if ratio < 1.0 else "CLOSE" if ratio < 1.1 else "WORSE"
        print(f"  {r['model']:<35s}: {ratio:.3f}x baseline ({status})")

    return d


def main():
    probe = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].startswith("--probe") else None
    specific = sys.argv[2] if len(sys.argv) > 2 else None

    if specific == "a" or probe is None:
        analyze_probe_a()
        print()
    if specific == "e" or probe is None:
        analyze_probe_e()
        print()
    if specific == "f" or probe is None:
        analyze_probe_f()
        print()
    if specific == "mvp" or probe is None:
        analyze_mvp()


if __name__ == "__main__":
    main()
