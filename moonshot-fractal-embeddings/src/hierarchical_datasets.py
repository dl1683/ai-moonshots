"""
Hierarchical Datasets for Fractal Embedding Training
=====================================================

Datasets with multi-level category hierarchies:
- Level 0: Coarse (e.g., Science, Arts, Sports)
- Level 1: Medium (e.g., Physics, Music, Football)
- Level 2: Fine (e.g., Quantum Mechanics, Jazz, NFL)
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import random

# Try to import datasets
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


@dataclass
class HierarchicalSample:
    """A sample with hierarchical labels."""
    text: str
    level0_label: int  # Coarsest
    level1_label: int  # Medium
    level2_label: int  # Finest
    level0_name: str
    level1_name: str
    level2_name: str


class HierarchicalDataset:
    """Base class for hierarchical datasets."""

    def __init__(self):
        self.samples: List[HierarchicalSample] = []
        self.level0_names: List[str] = []
        self.level1_names: List[str] = []
        self.level2_names: List[str] = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> HierarchicalSample:
        return self.samples[idx]

    def get_batch(self, batch_size: int) -> List[HierarchicalSample]:
        indices = np.random.choice(len(self.samples), batch_size, replace=False)
        return [self.samples[i] for i in indices]

    def get_hierarchy_stats(self) -> Dict:
        return {
            'num_samples': len(self.samples),
            'num_level0': len(self.level0_names),
            'num_level1': len(self.level1_names),
            'num_level2': len(self.level2_names),
            'level0_names': self.level0_names[:10],
            'level1_names': self.level1_names[:10],
        }


class AGNewsHierarchical(HierarchicalDataset):
    """
    AG News with synthetic sub-categories.

    Original: 4 categories (World, Sports, Business, Sci/Tech)
    We create sub-categories based on keywords.
    """

    def __init__(self, split: str = "train", max_samples: int = None):
        super().__init__()

        if not DATASETS_AVAILABLE:
            print("datasets library not available, using mock data")
            self._create_mock_data()
            return

        print("Loading AG News...")
        dataset = load_dataset("ag_news", split=split)

        # Original categories
        self.level0_names = ["World", "Sports", "Business", "Sci/Tech"]

        # Sub-categories (heuristic based on keywords)
        self.subcategories = {
            0: {  # World
                "Politics": ["president", "government", "election", "minister", "parliament"],
                "Conflict": ["war", "military", "attack", "troops", "bomb"],
                "Diplomacy": ["treaty", "agreement", "talks", "summit", "relations"],
                "Other World": [],
            },
            1: {  # Sports
                "Football": ["football", "nfl", "touchdown", "quarterback", "super bowl"],
                "Basketball": ["basketball", "nba", "lakers", "points", "rebounds"],
                "Baseball": ["baseball", "mlb", "yankees", "home run", "pitcher"],
                "Soccer": ["soccer", "fifa", "goal", "manchester", "champions league"],
                "Other Sports": [],
            },
            2: {  # Business
                "Markets": ["stock", "market", "dow", "nasdaq", "shares"],
                "Companies": ["company", "ceo", "merger", "acquisition", "profit"],
                "Economy": ["economy", "gdp", "inflation", "unemployment", "fed"],
                "Other Business": [],
            },
            3: {  # Sci/Tech
                "Software": ["software", "microsoft", "google", "app", "program"],
                "Hardware": ["computer", "chip", "processor", "device", "phone"],
                "Internet": ["internet", "online", "website", "web", "digital"],
                "Science": ["research", "study", "scientists", "discovery", "space"],
                "Other Tech": [],
            },
        }

        # Build level1 and level2 name lists
        self.level1_names = []
        self.level1_to_level0 = {}
        level1_id = 0
        for level0, subcats in self.subcategories.items():
            for subcat_name in subcats.keys():
                self.level1_names.append(subcat_name)
                self.level1_to_level0[level1_id] = level0
                level1_id += 1

        # Level 2 is individual samples (or clusters within subcategories)
        self.level2_names = []  # Will be populated per sample

        # Process samples
        samples_to_process = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))

        for i, item in enumerate(samples_to_process):
            text = item['text']
            level0 = item['label']
            level0_name = self.level0_names[level0]

            # Determine subcategory
            level1, level1_name = self._get_subcategory(text, level0)

            # Level 2 is the sample index within the subcategory
            level2 = i
            level2_name = f"sample_{i}"

            self.samples.append(HierarchicalSample(
                text=text,
                level0_label=level0,
                level1_label=level1,
                level2_label=level2,
                level0_name=level0_name,
                level1_name=level1_name,
                level2_name=level2_name,
            ))
            self.level2_names.append(level2_name)

        print(f"Loaded {len(self.samples)} samples")
        print(f"Hierarchy: {len(self.level0_names)} L0 -> {len(self.level1_names)} L1")

    def _get_subcategory(self, text: str, level0: int) -> Tuple[int, str]:
        """Determine subcategory based on keywords."""
        text_lower = text.lower()
        subcats = self.subcategories[level0]

        # Find matching subcategory
        base_level1 = sum(len(self.subcategories[i]) for i in range(level0))

        for i, (subcat_name, keywords) in enumerate(subcats.items()):
            if keywords:  # Skip "Other" category initially
                for keyword in keywords:
                    if keyword in text_lower:
                        return base_level1 + i, subcat_name

        # Default to "Other" category
        other_idx = len(subcats) - 1
        other_name = list(subcats.keys())[other_idx]
        return base_level1 + other_idx, other_name

    def _create_mock_data(self):
        """Create mock hierarchical data for testing."""
        self.level0_names = ["Science", "Sports", "Business", "Tech"]
        self.level1_names = [
            "Physics", "Biology", "Chemistry",
            "Football", "Basketball", "Soccer",
            "Finance", "Marketing", "HR",
            "Software", "Hardware", "AI"
        ]

        templates = {
            ("Science", "Physics"): [
                "The electron orbits around the nucleus in quantum mechanics.",
                "Einstein's theory of relativity changed our understanding of space and time.",
                "Gravity is a fundamental force described by general relativity.",
            ],
            ("Science", "Biology"): [
                "DNA contains the genetic instructions for all living organisms.",
                "Cells divide through mitosis to create new cells.",
                "Evolution drives the adaptation of species over time.",
            ],
            ("Sports", "Football"): [
                "The quarterback threw a touchdown pass in the fourth quarter.",
                "The defense stopped the running back at the goal line.",
                "The team won the Super Bowl championship.",
            ],
            ("Sports", "Basketball"): [
                "The point guard dribbled down the court for a layup.",
                "The center blocked the shot and grabbed the rebound.",
                "Free throws decided the close game.",
            ],
            ("Tech", "Software"): [
                "The new software update includes bug fixes and improvements.",
                "Developers use version control to manage code changes.",
                "The application crashed due to a memory leak.",
            ],
            ("Tech", "AI"): [
                "Machine learning models are trained on large datasets.",
                "Neural networks can recognize patterns in images.",
                "GPT models generate human-like text responses.",
            ],
        }

        level1_to_level0 = {
            "Physics": 0, "Biology": 0, "Chemistry": 0,
            "Football": 1, "Basketball": 1, "Soccer": 1,
            "Finance": 2, "Marketing": 2, "HR": 2,
            "Software": 3, "Hardware": 3, "AI": 3,
        }

        sample_id = 0
        for (l0_name, l1_name), texts in templates.items():
            l0 = self.level0_names.index(l0_name)
            l1 = self.level1_names.index(l1_name)

            for text in texts:
                self.samples.append(HierarchicalSample(
                    text=text,
                    level0_label=l0,
                    level1_label=l1,
                    level2_label=sample_id,
                    level0_name=l0_name,
                    level1_name=l1_name,
                    level2_name=f"sample_{sample_id}",
                ))
                self.level2_names.append(f"sample_{sample_id}")
                sample_id += 1


class DBPediaHierarchical(HierarchicalDataset):
    """
    DBPedia with natural hierarchy.

    14 categories with natural groupings:
    - Company, EducationalInstitution, Artist, Athlete, OfficeHolder,
    - MeanOfTransportation, Building, NaturalPlace, Village, Animal,
    - Plant, Album, Film, WrittenWork
    """

    def __init__(self, split: str = "train", max_samples: int = None):
        super().__init__()

        if not DATASETS_AVAILABLE:
            print("datasets library not available, using mock data")
            self._create_mock_data()
            return

        print("Loading DBPedia...")
        dataset = load_dataset("dbpedia_14", split=split)

        # Original 14 categories
        original_labels = [
            "Company", "EducationalInstitution", "Artist", "Athlete",
            "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace",
            "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"
        ]

        # Group into super-categories
        self.hierarchy = {
            "Organization": ["Company", "EducationalInstitution"],
            "Person": ["Artist", "Athlete", "OfficeHolder"],
            "Place": ["Building", "NaturalPlace", "Village"],
            "Living Thing": ["Animal", "Plant"],
            "Creative Work": ["Album", "Film", "WrittenWork"],
            "Object": ["MeanOfTransportation"],
        }

        # Build mappings
        self.level0_names = list(self.hierarchy.keys())
        self.level1_names = original_labels

        label_to_level0 = {}
        label_to_level1 = {}
        for l0_idx, (l0_name, l1_list) in enumerate(self.hierarchy.items()):
            for l1_name in l1_list:
                l1_idx = original_labels.index(l1_name)
                label_to_level0[l1_idx] = l0_idx
                label_to_level1[l1_idx] = self.level1_names.index(l1_name)

        # Process samples
        samples_to_process = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))

        for i, item in enumerate(samples_to_process):
            text = item['content']
            original_label = item['label']

            level0 = label_to_level0[original_label]
            level1 = label_to_level1[original_label]
            level2 = i

            self.samples.append(HierarchicalSample(
                text=text,
                level0_label=level0,
                level1_label=level1,
                level2_label=level2,
                level0_name=self.level0_names[level0],
                level1_name=self.level1_names[original_label],
                level2_name=f"sample_{i}",
            ))
            self.level2_names.append(f"sample_{i}")

        print(f"Loaded {len(self.samples)} samples")
        print(f"Hierarchy: {len(self.level0_names)} L0 -> {len(self.level1_names)} L1")

    def _create_mock_data(self):
        """Create mock data."""
        AGNewsHierarchical._create_mock_data(self)


class YahooAnswersHierarchical(HierarchicalDataset):
    """
    Yahoo Answers with topic hierarchy.

    10 categories that can be grouped.
    """

    def __init__(self, split: str = "train", max_samples: int = None):
        super().__init__()

        if not DATASETS_AVAILABLE:
            print("datasets library not available")
            self._create_mock_data()
            return

        print("Loading Yahoo Answers...")
        try:
            dataset = load_dataset("yahoo_answers_topics", split=split)
        except Exception as e:
            print(f"Could not load Yahoo Answers: {e}")
            self._create_mock_data()
            return

        # Original 10 categories
        original_labels = [
            "Society & Culture", "Science & Mathematics", "Health",
            "Education & Reference", "Computers & Internet", "Sports",
            "Business & Finance", "Entertainment & Music",
            "Family & Relationships", "Politics & Government"
        ]

        # Group into super-categories
        self.hierarchy = {
            "Knowledge": ["Science & Mathematics", "Education & Reference", "Computers & Internet"],
            "Lifestyle": ["Health", "Family & Relationships", "Society & Culture"],
            "Entertainment": ["Sports", "Entertainment & Music"],
            "Professional": ["Business & Finance", "Politics & Government"],
        }

        # Build mappings
        self.level0_names = list(self.hierarchy.keys())
        self.level1_names = original_labels

        label_to_level0 = {}
        for l0_idx, (l0_name, l1_list) in enumerate(self.hierarchy.items()):
            for l1_name in l1_list:
                l1_idx = original_labels.index(l1_name)
                label_to_level0[l1_idx] = l0_idx

        # Process samples
        samples_to_process = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))

        for i, item in enumerate(samples_to_process):
            # Yahoo Answers has question_title, question_content, best_answer
            text = f"{item['question_title']} {item['question_content']}"
            original_label = item['topic']

            level0 = label_to_level0[original_label]
            level1 = original_label
            level2 = i

            self.samples.append(HierarchicalSample(
                text=text,
                level0_label=level0,
                level1_label=level1,
                level2_label=level2,
                level0_name=self.level0_names[level0],
                level1_name=self.level1_names[original_label],
                level2_name=f"sample_{i}",
            ))
            self.level2_names.append(f"sample_{i}")

        print(f"Loaded {len(self.samples)} samples")

    def _create_mock_data(self):
        AGNewsHierarchical._create_mock_data(self)


class AmazonHierarchical(HierarchicalDataset):
    """
    Amazon reviews with product category hierarchy.
    """

    def __init__(self, split: str = "train", max_samples: int = 10000):
        super().__init__()

        if not DATASETS_AVAILABLE:
            self._create_mock_data()
            return

        print("Loading Amazon Reviews...")
        try:
            # Try loading a subset
            dataset = load_dataset("amazon_polarity", split=split)
        except Exception as e:
            print(f"Could not load Amazon: {e}")
            self._create_mock_data()
            return

        # Amazon polarity doesn't have categories, so we'll use keywords
        # This is a simplified version
        self.level0_names = ["Electronics", "Books", "Home", "Fashion"]
        self.level1_names = [
            "Phones", "Computers", "Audio",  # Electronics
            "Fiction", "Non-Fiction", "Technical",  # Books
            "Kitchen", "Furniture", "Decor",  # Home
            "Clothing", "Shoes", "Accessories",  # Fashion
        ]

        # Keyword mappings
        keywords = {
            "Phones": ["phone", "iphone", "android", "samsung", "mobile"],
            "Computers": ["laptop", "computer", "pc", "macbook", "desktop"],
            "Audio": ["headphones", "speaker", "audio", "earbuds", "sound"],
            "Fiction": ["novel", "story", "fiction", "characters", "plot"],
            "Non-Fiction": ["biography", "history", "memoir", "true"],
            "Technical": ["programming", "science", "textbook", "guide"],
            "Kitchen": ["kitchen", "cooking", "pan", "knife", "appliance"],
            "Furniture": ["chair", "table", "desk", "bed", "sofa"],
            "Decor": ["decor", "decoration", "art", "frame", "lamp"],
            "Clothing": ["shirt", "dress", "pants", "jacket", "clothing"],
            "Shoes": ["shoes", "boots", "sneakers", "sandals", "footwear"],
            "Accessories": ["watch", "bag", "wallet", "jewelry", "belt"],
        }

        level1_to_level0 = {
            "Phones": 0, "Computers": 0, "Audio": 0,
            "Fiction": 1, "Non-Fiction": 1, "Technical": 1,
            "Kitchen": 2, "Furniture": 2, "Decor": 2,
            "Clothing": 3, "Shoes": 3, "Accessories": 3,
        }

        # Process samples
        samples_to_process = dataset.select(range(min(max_samples, len(dataset))))

        for i, item in enumerate(samples_to_process):
            text = item['content']
            text_lower = text.lower()

            # Determine category from keywords
            level1_name = "Phones"  # Default
            for cat, kws in keywords.items():
                if any(kw in text_lower for kw in kws):
                    level1_name = cat
                    break

            level0 = level1_to_level0[level1_name]
            level1 = self.level1_names.index(level1_name)
            level2 = i

            self.samples.append(HierarchicalSample(
                text=text,
                level0_label=level0,
                level1_label=level1,
                level2_label=level2,
                level0_name=self.level0_names[level0],
                level1_name=level1_name,
                level2_name=f"sample_{i}",
            ))
            self.level2_names.append(f"sample_{i}")

        print(f"Loaded {len(self.samples)} samples")

    def _create_mock_data(self):
        AGNewsHierarchical._create_mock_data(self)


class CLINCHierarchical(HierarchicalDataset):
    """
    CLINC150 intent detection dataset with natural two-level hierarchy.

    L0 = domain (10 domains, excluding oos), L1 = intent (150 intents).
    HF ID: contemmcm/clinc150  (single 'complete' split with a 'split' column)
    """

    def __init__(self, split: str = "train", max_samples: int = None):
        super().__init__()

        if not DATASETS_AVAILABLE:
            print("datasets library not available")
            self._create_mock_data()
            return

        print("Loading CLINC150...")
        try:
            # Dataset has a single 'complete' split; filter by the 'split' column
            dataset = load_dataset("contemmcm/clinc150", split="complete")
        except Exception as e:
            print(f"Could not load CLINC150: {e}")
            self._create_mock_data()
            return

        # Map split names: "train" -> "train", "test" -> "test", "val" -> "val"
        split_name = "val" if split == "validation" else split
        dataset = dataset.filter(lambda x: x["split"] == split_name)

        # Get ClassLabel features for domain and intent
        domain_feature = dataset.features["domain"]
        intent_feature = dataset.features["intent"]
        domain_names = domain_feature.names  # includes 'oos' at index 0
        intent_names = intent_feature.names  # includes 'oos:oos' at index 0

        # Filter out oos domain (index 0)
        dataset = dataset.filter(lambda x: x["domain"] != 0)

        # Build non-oos domain/intent name lists (contiguous re-indexing)
        non_oos_domains = [d for d in domain_names if d != "oos"]
        non_oos_intents = [i for i in intent_names if not i.startswith("oos")]
        self.level0_names = non_oos_domains
        self.level1_names = non_oos_intents
        self.level2_names = []

        # Map original ClassLabel int -> new contiguous index
        domain_remap = {}
        for orig_idx, name in enumerate(domain_names):
            if name in non_oos_domains:
                domain_remap[orig_idx] = non_oos_domains.index(name)
        intent_remap = {}
        for orig_idx, name in enumerate(intent_names):
            if name in non_oos_intents:
                intent_remap[orig_idx] = non_oos_intents.index(name)

        # Process samples
        items = list(dataset)
        if max_samples is not None:
            items = items[:max_samples]

        for i, item in enumerate(items):
            text = item["text"]
            level0 = domain_remap.get(item["domain"])
            level1 = intent_remap.get(item["intent"])
            if level0 is None or level1 is None:
                continue

            self.samples.append(HierarchicalSample(
                text=text,
                level0_label=level0,
                level1_label=level1,
                level2_label=i,
                level0_name=self.level0_names[level0],
                level1_name=self.level1_names[level1],
                level2_name=f"sample_{i}",
            ))
            self.level2_names.append(f"sample_{i}")

        print(f"Loaded {len(self.samples)} samples")
        print(f"Hierarchy: {len(self.level0_names)} L0 -> {len(self.level1_names)} L1")

    def _create_mock_data(self):
        AGNewsHierarchical._create_mock_data(self)


class TRECHierarchical(HierarchicalDataset):
    """
    TREC question classification with native coarse/fine hierarchy.

    L0 = 6 coarse question types, L1 = 50 fine question types.
    HF ID: SetFit/TREC-QC  (has label_coarse and label columns with text names)
    """

    def __init__(self, split: str = "train", max_samples: int = None):
        super().__init__()

        if not DATASETS_AVAILABLE:
            print("datasets library not available")
            self._create_mock_data()
            return

        print("Loading TREC...")
        try:
            dataset = load_dataset("SetFit/TREC-QC", split=split)
        except Exception as e:
            print(f"Could not load TREC: {e}")
            self._create_mock_data()
            return

        # Columns: text, label (fine int), label_text, label_coarse (coarse int),
        #          label_coarse_text, label_original (e.g. "DESC:manner")
        # Build name lists from BOTH train and test to ensure complete coverage
        coarse_set = {}
        fine_set = {}
        try:
            full_ds = load_dataset("SetFit/TREC-QC")
            for s in full_ds.values():
                for item in s:
                    coarse_set[item["label_coarse"]] = item["label_coarse_text"]
                    fine_set[item["label"]] = item["label_text"]
        except Exception:
            for item in dataset:
                coarse_set[item["label_coarse"]] = item["label_coarse_text"]
                fine_set[item["label"]] = item["label_text"]

        self.level0_names = [coarse_set[k] for k in sorted(coarse_set.keys())]
        self.level1_names = [fine_set[k] for k in sorted(fine_set.keys())]
        self.level2_names = []

        # Process samples
        items = list(dataset)
        if max_samples is not None:
            items = items[:max_samples]

        for i, item in enumerate(items):
            text = item["text"]
            level0 = item["label_coarse"]
            level1 = item["label"]

            self.samples.append(HierarchicalSample(
                text=text,
                level0_label=level0,
                level1_label=level1,
                level2_label=i,
                level0_name=self.level0_names[level0],
                level1_name=self.level1_names[level1],
                level2_name=f"sample_{i}",
            ))
            self.level2_names.append(f"sample_{i}")

        print(f"Loaded {len(self.samples)} samples")
        print(f"Hierarchy: {len(self.level0_names)} L0 -> {len(self.level1_names)} L1")

    def _create_mock_data(self):
        AGNewsHierarchical._create_mock_data(self)


def load_hierarchical_dataset(name: str, split: str = "train", max_samples: int = None) -> HierarchicalDataset:
    """Load a hierarchical dataset by name."""
    datasets_map = {
        "agnews": AGNewsHierarchical,
        "dbpedia": DBPediaHierarchical,
        "yahoo": YahooAnswersHierarchical,
        "amazon": AmazonHierarchical,
        "clinc": CLINCHierarchical,
        "trec": TRECHierarchical,
    }

    if name.lower() not in datasets_map:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(datasets_map.keys())}")

    return datasets_map[name.lower()](split=split, max_samples=max_samples)


# =============================================================================
# TEST
# =============================================================================

def test_hierarchical_datasets():
    """Test loading hierarchical datasets."""
    print("=" * 60)
    print("Testing Hierarchical Datasets")
    print("=" * 60)

    for name in ["agnews", "dbpedia"]:
        print(f"\n--- {name.upper()} ---")
        try:
            dataset = load_hierarchical_dataset(name, split="train", max_samples=1000)
            stats = dataset.get_hierarchy_stats()
            print(f"Samples: {stats['num_samples']}")
            print(f"Level 0: {stats['num_level0']} categories")
            print(f"Level 1: {stats['num_level1']} categories")
            print(f"L0 names: {stats['level0_names']}")
            print(f"L1 names: {stats['level1_names']}")

            # Show sample
            sample = dataset[0]
            print(f"\nSample:")
            print(f"  Text: {sample.text[:100]}...")
            print(f"  L0: {sample.level0_name} ({sample.level0_label})")
            print(f"  L1: {sample.level1_name} ({sample.level1_label})")
        except Exception as e:
            print(f"Error: {e}")

    print("\nHierarchical datasets test complete!")


if __name__ == "__main__":
    test_hierarchical_datasets()
