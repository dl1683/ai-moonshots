#!/bin/bash
# Multi-Codex review at eval checkpoint
# Run this after every eval step (every 5000 steps)
# Usage: bash code/eval_checkpoint_review.sh <step_number>

STEP=${1:-5000}
REPO="C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra"

echo "=== MULTI-CODEX REVIEW @ STEP $STEP ==="
echo "Launching 4 parallel Codex reviews..."

# 1. Code Reviewer
codex exec -s read-only --skip-git-repo-check -C "$REPO" \
  -o "results/review_step${STEP}_code.md" \
  "Read CLAUDE.md. Read code/sutra_v05_ssm.py and code/sutra_v05_train.py.
   Quick code review: any new bugs or regressions since last checkpoint?
   Check loss trajectory in results/v05_log.txt for anomalies.
   Is training healthy? Any red flags? Brief answer." &

# 2. Chrome Experimenter
codex exec -s read-only --skip-git-repo-check -C "$REPO" \
  -o "results/review_step${STEP}_chrome.md" \
  "Read CLAUDE.md and research/RESEARCH.md (Chrome probe sections).
   Given training at step $STEP, what are the 3 most useful CPU
   experiments to run RIGHT NOW? Give specific hypotheses and kill criteria." &

# 3. Competitive Analyst
codex exec -s read-only --skip-git-repo-check -C "$REPO" \
  -o "results/review_step${STEP}_competitive.md" \
  "Read CLAUDE.md. Check results/v05_metrics.json for latest eval.
   Compare to Pythia-70m (BPT 3.559 on clean benchmark).
   How far are we? What closes the gap? Brief quantitative answer." &

# 4. Architecture Designer
codex exec -s read-only --skip-git-repo-check -C "$REPO" \
  -o "results/review_step${STEP}_architecture.md" \
  "Read CLAUDE.md, research/VISION.md, code/sutra_v05_ssm.py.
   Based on current training progress, should we change anything
   for the next phase? Any architecture modifications worth testing?
   Focus on what makes the model GENERATE BETTER TEXT." &

echo "All 4 reviews launched. Results in results/review_step${STEP}_*.md"
wait
echo "All reviews complete."
