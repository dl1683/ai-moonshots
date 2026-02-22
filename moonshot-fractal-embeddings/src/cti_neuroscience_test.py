"""
CTI Universal Law: Test on BIOLOGICAL NEURONS (calcium imaging, mouse V1).

Dataset: DANDI:000039 - Allen Institute contrast tuning in mouse visual cortex
Data: Two-photon calcium imaging, 8 grating orientations, ~200 neurons
Test: Does logit(q_norm) = alpha * kappa_nearest + C hold for neural population codes?

If yes: the same EVT-derived law governs BOTH silicon and carbon-based neural codes.
This would be the most impactful finding.
"""
import numpy as np
from scipy.stats import pearsonr, linregress
from scipy.special import logit
import json
import time
import warnings
warnings.filterwarnings('ignore')

print("=== CTI Universal Law: Biological Neural Test ===")
print("Dataset: DANDI:000039 (Allen Institute, mouse V1, drifting gratings)")
print()

# --- Download first session from DANDI ---
print("Connecting to DANDI archive...")
from dandi.dandiapi import DandiAPIClient

client = DandiAPIClient()
ds = client.get_dandiset('000039')
assets = list(ds.get_assets())
print(f"Found {len(assets)} sessions")

# Use first session
asset = assets[0]
print(f"Session: {asset.path}, size: {asset.size/1e6:.1f} MB")

# Download the file
import os
CACHE_PATH = 'data/neuro_session.nwb'
if not os.path.exists(CACHE_PATH):
    os.makedirs('data', exist_ok=True)
    print(f"Downloading {asset.size/1e6:.1f} MB... (this takes ~60s)")
    t0 = time.time()
    asset.download(CACHE_PATH)
    print(f"Downloaded in {time.time()-t0:.1f}s")
else:
    print(f"Using cached: {CACHE_PATH}")

# --- Load NWB and extract neural responses ---
print("\nLoading NWB file...")
import pynwb
io = pynwb.NWBHDF5IO(CACHE_PATH, 'r')
nwb = io.read()
print(f"Subject: {nwb.subject.description if hasattr(nwb, 'subject') and nwb.subject else 'unknown'}")

# Find fluorescence data
try:
    fluorescence = nwb.acquisition['ophys']['Fluorescence']['RoiResponseSeries']
    print(f"Fluorescence shape: {fluorescence.data.shape}")
    F = fluorescence.data[:]  # [T, N_neurons]
except:
    # Try different path
    try:
        ophys = nwb.processing['ophys']
        fluor_module = ophys.data_interfaces['Fluorescence']
        roi_series = fluor_module.roi_response_series['RoiResponseSeries']
        F = roi_series.data[:]
        print(f"Fluorescence shape: {F.shape}")
    except Exception as e:
        print(f"Error accessing fluorescence: {e}")
        print("Available acquisition keys:", list(nwb.acquisition.keys()))
        print("Available processing keys:", list(nwb.processing.keys()))
        io.close()
        exit(1)

# Find stimulus presentations
print("\nFinding stimulus presentations...")
intervals = nwb.intervals
print("Available intervals:", list(intervals.keys()))

# Look for gratings stimuli
gratings_key = None
for key in intervals.keys():
    if 'grat' in key.lower() or 'stim' in key.lower():
        gratings_key = key
        print(f"Found: {key}")
        break

if gratings_key is None:
    # Try trials
    if 'trials' in dir(nwb) and nwb.trials is not None:
        trials_df = nwb.trials.to_dataframe()
        print(f"Trials columns: {list(trials_df.columns)}")
        print(trials_df.head())
    io.close()
    exit(1)

# Get stimulus table
stim_df = intervals[gratings_key].to_dataframe()
print(f"\nStimulus table shape: {stim_df.shape}")
print(f"Columns: {list(stim_df.columns)}")
print(stim_df.head())

io.close()
