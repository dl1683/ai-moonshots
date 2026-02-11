"""
Deep Hierarchy Datasets (4+ levels)
====================================

Loaders for naturally deep hierarchical text datasets.
These extend the existing 2-level interface (L0/L1) but can
be configured to target any pair of hierarchy levels.

Datasets:
1. HUPD (USPTO Patents + CPC codes) — 5 levels
2. ENZYME (PubMed + EC numbers) — 4 levels

Usage:
    ds = load_deep_hierarchy_dataset(
        "hupd", split="train", max_samples=10000,
        coarse_level=0,  # Section (9 classes)
        fine_level=2,     # Subclass (~700 classes)
    )
"""

import sys
import os
import json
import re
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(__file__))

from hierarchical_datasets import HierarchicalSample, HierarchicalDataset

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


# ============================================================
# CPC Code Parsing
# ============================================================

def parse_cpc_code(cpc_str: str) -> Dict[str, str]:
    """Parse a CPC code into its hierarchy levels.

    CPC format: A01B1/02
    - Section: A (1 letter)
    - Class: A01 (letter + 2 digits)
    - Subclass: A01B (letter + 2 digits + letter)
    - Main Group: A01B1 or A01B1/00 (before /)
    - Subgroup: A01B1/02 (after /)

    Returns dict with keys: section, cls, subclass, group, subgroup
    """
    cpc = cpc_str.strip()
    if not cpc or len(cpc) < 4:
        return None

    result = {}

    # Section: first letter
    result['section'] = cpc[0]

    # Class: first 3 chars (letter + 2 digits)
    if len(cpc) >= 3:
        result['cls'] = cpc[:3]
    else:
        return None

    # Subclass: first 4 chars (letter + 2 digits + letter)
    if len(cpc) >= 4:
        result['subclass'] = cpc[:4]
    else:
        return None

    # Group and subgroup: parse the rest
    # After subclass letter, there may be digits, then / then digits
    remainder = cpc[4:]
    if '/' in remainder:
        parts = remainder.split('/')
        result['group'] = cpc[:4] + parts[0].strip()
        result['subgroup'] = cpc.replace(' ', '')
    elif remainder.strip():
        result['group'] = cpc[:4] + remainder.strip()
        result['subgroup'] = cpc.replace(' ', '')
    else:
        result['group'] = cpc[:4]
        result['subgroup'] = cpc[:4]

    return result


CPC_LEVEL_NAMES = ['section', 'cls', 'subclass', 'group', 'subgroup']
CPC_LEVEL_DESCRIPTIONS = {
    'section': 'CPC Section (A-H, Y)',
    'cls': 'CPC Class (e.g., A01)',
    'subclass': 'CPC Subclass (e.g., A01B)',
    'group': 'CPC Main Group (e.g., A01B1)',
    'subgroup': 'CPC Subgroup (e.g., A01B1/02)',
}


# ============================================================
# HUPD Dataset (USPTO Patents + CPC Hierarchy)
# ============================================================

class HUPDHierarchical(HierarchicalDataset):
    """Harvard USPTO Patent Dataset with CPC hierarchy.

    3-level usable hierarchy: Section (9) -> Class (125) -> Subclass (626)
    Text: invention_title (patent titles, 10-20 words avg)
    Data: 1.9M patents with CPC codes from 4.5M total

    Loads metadata via pandas from HuggingFace hub feather file.
    No abstract available in metadata — uses titles only.

    Args:
        split: "train" or "test" (80/20 chronological split)
        max_samples: limit samples
        coarse_level: 0=section(9), 1=class(125), 2=subclass(626)
        fine_level: must be > coarse_level
        min_samples_per_class: filter rare classes
    """

    FEATHER_PATH = None  # Cached path

    def __init__(
        self,
        split: str = "train",
        max_samples: int = None,
        coarse_level: int = 0,
        fine_level: int = 2,
        min_samples_per_class: int = 10,
    ):
        super().__init__()
        self.coarse_level = coarse_level
        self.fine_level = fine_level
        # Map levels: 0=section, 1=class, 2=subclass
        level_names = ['section', 'cls', 'subclass']
        level_chars = [1, 3, 4]  # How many chars of CPC code for each level

        coarse_key = level_names[coarse_level]
        fine_key = level_names[fine_level]
        coarse_chars = level_chars[coarse_level]
        fine_chars = level_chars[fine_level]

        print(f"Loading HUPD patents...")
        print(f"  Coarse: {coarse_key} ({coarse_chars} chars)")
        print(f"  Fine: {fine_key} ({fine_chars} chars)")

        # Download feather file via huggingface_hub
        import pandas as pd
        from huggingface_hub import hf_hub_download

        if HUPDHierarchical.FEATHER_PATH is None:
            HUPDHierarchical.FEATHER_PATH = hf_hub_download(
                repo_id='HUPD/hupd',
                filename='hupd_metadata_2022-02-22.feather',
                repo_type='dataset',
            )
        feather_path = HUPDHierarchical.FEATHER_PATH

        df = pd.read_feather(feather_path)
        # Filter to patents with CPC codes and titles
        df = df[df['main_cpc_label'].str.len() > 0].copy()
        df = df[df['invention_title'].notna() & (df['invention_title'].str.len() > 10)].copy()

        # Parse hierarchy from CPC code prefix
        df['coarse'] = df['main_cpc_label'].str[:coarse_chars]
        df['fine'] = df['main_cpc_label'].str[:fine_chars]

        # Chronological train/test split (80/20)
        df = df.sort_values('filing_date')
        split_idx = int(len(df) * 0.8)
        if split == "train":
            df = df.iloc[:split_idx]
        else:
            df = df.iloc[split_idx:]

        # Filter rare classes
        fine_counts = df['fine'].value_counts()
        valid_fines = set(fine_counts[fine_counts >= min_samples_per_class].index)
        df = df[df['fine'].isin(valid_fines)]

        # Also filter coarse classes with too few samples
        coarse_counts = df['coarse'].value_counts()
        valid_coarse = set(coarse_counts[coarse_counts >= min_samples_per_class].index)
        df = df[df['coarse'].isin(valid_coarse)]

        # Build label mappings
        coarse_set = sorted(df['coarse'].unique())
        fine_set = sorted(df['fine'].unique())
        coarse_to_id = {name: i for i, name in enumerate(coarse_set)}
        fine_to_id = {name: i for i, name in enumerate(fine_set)}

        self.level0_names = coarse_set
        self.level1_names = fine_set

        # Sample if needed
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)

        # Build samples
        for _, row in df.iterrows():
            coarse_name = row['coarse']
            fine_name = row['fine']
            if coarse_name not in coarse_to_id or fine_name not in fine_to_id:
                continue
            self.samples.append(HierarchicalSample(
                text=str(row['invention_title'])[:512],
                level0_label=coarse_to_id[coarse_name],
                level1_label=fine_to_id[fine_name],
                level2_label=0,
                level0_name=coarse_name,
                level1_name=fine_name,
                level2_name='',
            ))

        print(f"  Loaded {len(self.samples)} patents ({split})")
        print(f"  L0 classes ({coarse_key}): {len(self.level0_names)}")
        print(f"  L1 classes ({fine_key}): {len(self.level1_names)}")

        # Compute conditional entropy
        self._compute_entropy()

    def _compute_entropy(self):
        """Compute H(L1|L0) for this hierarchy configuration."""
        from math import log2
        l0_counts = Counter()
        joint_counts = Counter()

        for s in self.samples:
            l0_counts[s.level0_label] += 1
            joint_counts[(s.level0_label, s.level1_label)] += 1

        n = len(self.samples)
        h_l1_given_l0 = 0.0

        for l0 in l0_counts:
            p_l0 = l0_counts[l0] / n
            # H(L1|L0=l0)
            h_l1_l0 = 0.0
            for (l0_, l1), count in joint_counts.items():
                if l0_ != l0:
                    continue
                p_l1_given_l0 = count / l0_counts[l0]
                if p_l1_given_l0 > 0:
                    h_l1_l0 -= p_l1_given_l0 * log2(p_l1_given_l0)
            h_l1_given_l0 += p_l0 * h_l1_l0

        self.conditional_entropy = h_l1_given_l0
        print(f"  H(L1|L0) = {h_l1_given_l0:.2f} bits")


# ============================================================
# ENZYME Dataset (PubMed + EC Number Hierarchy)
# ============================================================

def parse_ec_number(ec_str: str) -> Dict[str, str]:
    """Parse an EC number into hierarchy levels.

    EC format: 1.2.3.45
    - Level 0: First digit (7 main classes)
    - Level 1: First two digits (e.g., 1.2)
    - Level 2: First three digits (e.g., 1.2.3)
    - Level 3: Full EC number (e.g., 1.2.3.45)
    """
    parts = ec_str.strip().split('.')
    if len(parts) < 2:
        return None

    result = {}
    result['ec1'] = parts[0]
    result['ec2'] = '.'.join(parts[:2])
    if len(parts) >= 3:
        result['ec3'] = '.'.join(parts[:3])
    if len(parts) >= 4:
        result['ec4'] = '.'.join(parts[:4])

    return result


EC_LEVEL_NAMES = ['ec1', 'ec2', 'ec3', 'ec4']


class ENZYMEHierarchical(HierarchicalDataset):
    """ENZYME dataset from HiGen paper.

    4-level hierarchy based on EC (Enzyme Commission) numbers.
    Text: title + abstract from PubMed articles.

    Requires downloading from: https://github.com/viditjain99/HiGen

    Args:
        data_dir: path to extracted ENZYME dataset
        split: "train" or "test"
        max_samples: limit samples
        coarse_level: which EC level for L0 (0=ec1, 1=ec2, 2=ec3)
        fine_level: which EC level for L1 (must be > coarse_level)
    """

    def __init__(
        self,
        split: str = "train",
        max_samples: int = None,
        data_dir: str = None,
        coarse_level: int = 0,
        fine_level: int = 3,
        min_samples_per_class: int = 5,
    ):
        super().__init__()
        self.coarse_level = coarse_level
        self.fine_level = fine_level

        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data" / "enzyme"

        data_dir = Path(data_dir)

        if not data_dir.exists():
            print(f"ENZYME data not found at {data_dir}")
            print("Download from: https://github.com/viditjain99/HiGen")
            print("Attempting to download...")
            self._download(data_dir)

        self._load(data_dir, split, max_samples, coarse_level, fine_level, min_samples_per_class)

    def _download(self, data_dir: Path):
        """Try to download ENZYME dataset."""
        data_dir.mkdir(parents=True, exist_ok=True)
        try:
            import subprocess
            # Clone HiGen repo to get the data
            tmp_dir = data_dir.parent / "higen_tmp"
            if not tmp_dir.exists():
                subprocess.run(
                    ["git", "clone", "--depth=1",
                     "https://github.com/viditjain99/HiGen.git",
                     str(tmp_dir)],
                    check=True, capture_output=True,
                )
            # Look for ENZYME data in the repo
            import shutil
            for candidate in [tmp_dir / "data" / "ENZYME", tmp_dir / "ENZYME",
                              tmp_dir / "datasets" / "ENZYME"]:
                if candidate.exists():
                    shutil.copytree(candidate, data_dir, dirs_exist_ok=True)
                    print(f"  Copied ENZYME data from {candidate}")
                    return
            print("  Could not find ENZYME data in HiGen repo")
        except Exception as e:
            print(f"  Download failed: {e}")

    def _load(self, data_dir, split, max_samples, coarse_level, fine_level, min_samples_per_class):
        """Load and parse ENZYME data."""
        coarse_key = EC_LEVEL_NAMES[coarse_level]
        fine_key = EC_LEVEL_NAMES[fine_level]

        print(f"Loading ENZYME dataset...")
        print(f"  Coarse: {coarse_key}, Fine: {fine_key}")

        # Try multiple possible file formats
        data_file = None
        for candidate in [
            data_dir / f"{split}.json",
            data_dir / f"{split}.jsonl",
            data_dir / f"{split}_data.json",
            data_dir / f"enzyme_{split}.json",
        ]:
            if candidate.exists():
                data_file = candidate
                break

        if data_file is None:
            # Try loading all data from a single file
            for candidate in [
                data_dir / "data.json",
                data_dir / "enzyme.json",
                data_dir / "enzyme_data.json",
            ]:
                if candidate.exists():
                    data_file = candidate
                    break

        if data_file is None:
            print(f"  No data files found in {data_dir}")
            print(f"  Contents: {list(data_dir.iterdir()) if data_dir.exists() else 'dir not found'}")
            return

        print(f"  Loading from {data_file}")

        # Load data
        with open(data_file) as f:
            if data_file.suffix == '.jsonl':
                raw_data = [json.loads(line) for line in f if line.strip()]
            else:
                raw_data = json.load(f)
                if isinstance(raw_data, dict):
                    # May be keyed by split
                    raw_data = raw_data.get(split, raw_data.get('data', []))

        # Parse EC numbers and build samples
        parsed_data = []
        for item in raw_data:
            # Try different field names
            ec = item.get('ec_number', item.get('label', item.get('ec', '')))
            text = item.get('text', item.get('title', '') + '. ' + item.get('abstract', ''))

            if not ec or not text or len(text) < 20:
                continue

            levels = parse_ec_number(str(ec))
            if levels is None or coarse_key not in levels or fine_key not in levels:
                continue

            parsed_data.append({
                'text': text[:1024],
                'coarse': levels[coarse_key],
                'fine': levels[fine_key],
                'all_levels': levels,
            })

        # Filter rare classes
        fine_counts = Counter(d['fine'] for d in parsed_data)
        valid_fines = {f for f, c in fine_counts.items() if c >= min_samples_per_class}

        # Build label mappings
        coarse_set = sorted(set(d['coarse'] for d in parsed_data if d['fine'] in valid_fines))
        fine_set = sorted(set(d['fine'] for d in parsed_data if d['fine'] in valid_fines))

        coarse_to_id = {name: i for i, name in enumerate(coarse_set)}
        fine_to_id = {name: i for i, name in enumerate(fine_set)}

        self.level0_names = coarse_set
        self.level1_names = fine_set

        for d in parsed_data:
            if d['fine'] not in valid_fines:
                continue
            if d['coarse'] not in coarse_to_id or d['fine'] not in fine_to_id:
                continue

            self.samples.append(HierarchicalSample(
                text=d['text'],
                level0_label=coarse_to_id[d['coarse']],
                level1_label=fine_to_id[d['fine']],
                level2_label=0,
                level0_name=d['coarse'],
                level1_name=d['fine'],
                level2_name='',
            ))

        if max_samples and len(self.samples) > max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        print(f"  Loaded {len(self.samples)} articles")
        print(f"  L0 classes ({coarse_key}): {len(self.level0_names)}")
        print(f"  L1 classes ({fine_key}): {len(self.level1_names)}")


# ============================================================
# Amazon Reviews with Product Hierarchy
# ============================================================

class AmazonDeepHierarchical(HierarchicalDataset):
    """Amazon Reviews 2023 with product category hierarchy.

    Categories like: "Electronics > Computers & Accessories > Laptops > Gaming Laptops"
    Can provide 3-5 levels of hierarchy.

    Uses McAuley-Lab/Amazon-Reviews-2023 from HuggingFace.
    """

    def __init__(
        self,
        split: str = "train",
        max_samples: int = None,
        coarse_level: int = 0,
        fine_level: int = 2,
        min_samples_per_class: int = 10,
        category: str = "Electronics",
    ):
        super().__init__()

        if not DATASETS_AVAILABLE:
            raise RuntimeError("datasets library required")

        print(f"Loading Amazon Reviews 2023 ({category})...")

        # Load the dataset for a specific category
        try:
            ds = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023",
                f"raw_review_{category}",
                split=split if split == "train" else "test",
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"  Failed to load Amazon {category}: {e}")
            print("  Trying alternative loading...")
            # Try meta data which has categories
            try:
                ds = load_dataset(
                    "McAuley-Lab/Amazon-Reviews-2023",
                    f"raw_meta_{category}",
                    split="full",
                    trust_remote_code=True,
                )
            except Exception as e2:
                print(f"  Also failed: {e2}")
                return

        # Parse category hierarchies
        parsed_data = []
        for item in ds:
            # Categories field contains the hierarchy
            categories = item.get('categories', item.get('category', []))
            text = item.get('text', item.get('title', ''))

            if isinstance(categories, str):
                # May be delimited
                cats = [c.strip() for c in categories.split('>')]
            elif isinstance(categories, list):
                if categories and isinstance(categories[0], list):
                    cats = categories[0]  # Nested list
                else:
                    cats = categories
            else:
                continue

            if len(cats) <= max(coarse_level, fine_level):
                continue

            if not text or len(text) < 20:
                continue

            parsed_data.append({
                'text': text[:512],
                'coarse': cats[coarse_level],
                'fine': cats[fine_level],
                'all_cats': cats,
            })

            if max_samples and len(parsed_data) >= max_samples * 3:
                break  # Early stop for huge datasets

        # Filter and build (same as HUPD)
        fine_counts = Counter(d['fine'] for d in parsed_data)
        valid_fines = {f for f, c in fine_counts.items() if c >= min_samples_per_class}

        coarse_set = sorted(set(d['coarse'] for d in parsed_data if d['fine'] in valid_fines))
        fine_set = sorted(set(d['fine'] for d in parsed_data if d['fine'] in valid_fines))

        coarse_to_id = {name: i for i, name in enumerate(coarse_set)}
        fine_to_id = {name: i for i, name in enumerate(fine_set)}

        self.level0_names = coarse_set
        self.level1_names = fine_set

        for d in parsed_data:
            if d['fine'] not in valid_fines:
                continue
            if d['coarse'] not in coarse_to_id or d['fine'] not in fine_to_id:
                continue

            self.samples.append(HierarchicalSample(
                text=d['text'],
                level0_label=coarse_to_id[d['coarse']],
                level1_label=fine_to_id[d['fine']],
                level2_label=0,
                level0_name=d['coarse'],
                level1_name=d['fine'],
                level2_name='',
            ))

        if max_samples and len(self.samples) > max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        print(f"  Loaded {len(self.samples)} reviews")
        print(f"  L0 classes: {len(self.level0_names)}")
        print(f"  L1 classes: {len(self.level1_names)}")


# ============================================================
# Registry
# ============================================================

DEEP_DATASETS = {
    'hupd': HUPDHierarchical,
    'hupd_patents': HUPDHierarchical,
    'enzyme': ENZYMEHierarchical,
    'amazon_deep': AmazonDeepHierarchical,
}


def load_deep_hierarchy_dataset(
    name: str,
    split: str = "train",
    max_samples: int = None,
    coarse_level: int = 0,
    fine_level: int = 2,
    **kwargs,
) -> HierarchicalDataset:
    """Load a deep hierarchy dataset by name."""
    if name.lower() not in DEEP_DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DEEP_DATASETS.keys())}")

    cls = DEEP_DATASETS[name.lower()]
    return cls(
        split=split,
        max_samples=max_samples,
        coarse_level=coarse_level,
        fine_level=fine_level,
        **kwargs,
    )


# ============================================================
# Quick test
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hupd", choices=list(DEEP_DATASETS.keys()))
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--coarse-level", type=int, default=0)
    parser.add_argument("--fine-level", type=int, default=2)
    args = parser.parse_args()

    ds = load_deep_hierarchy_dataset(
        args.dataset,
        split="train",
        max_samples=args.max_samples,
        coarse_level=args.coarse_level,
        fine_level=args.fine_level,
    )

    stats = ds.get_hierarchy_stats()
    print(f"\nDataset stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if ds.samples:
        print(f"\nSample:")
        s = ds.samples[0]
        print(f"  Text: {s.text[:100]}...")
        print(f"  L0: {s.level0_name} (id={s.level0_label})")
        print(f"  L1: {s.level1_name} (id={s.level1_label})")
