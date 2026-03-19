#!/bin/bash
# Watch for probe results and touch a signal file when they appear
RESULTS="C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/results"
while true; do
  for f in probe_a_compression_capability.json probe_f_stigmergic.json; do
    if [ -f "$RESULTS/$f" ] && [ ! -f "$RESULTS/${f}.notified" ]; then
      echo "PROBE COMPLETE: $f" >> "$RESULTS/probe_notifications.txt"
      touch "$RESULTS/${f}.notified"
    fi
  done
  sleep 30
done
