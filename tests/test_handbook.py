import copy
import json
from pathlib import Path

import pytest

from scripts.validate_handbook import validate_handbook, validate_cases

ROOT = Path(__file__).resolve().parents[1]


def test_complete_handbook_and_frozen_cases():
    handbook = json.loads((ROOT / 'data/knowledge/handbook.json').read_text())
    cases = json.loads((ROOT / 'data/evaluation/assistant_cases.json').read_text())
    split = json.loads((ROOT / 'data/evaluation/assistant_split.json').read_text())
    validate_handbook(handbook)
    validate_cases(cases, split, handbook)


def test_reject_broken_source_and_duplicate_id():
    handbook = json.loads((ROOT / 'data/knowledge/handbook.json').read_text())
    bad = copy.deepcopy(handbook)
    bad['articles'][0]['related_ids'] = ['MISSING']
    with pytest.raises(ValueError, match='reference'):
        validate_handbook(bad)
    bad = copy.deepcopy(handbook)
    bad['articles'][1]['id'] = bad['articles'][0]['id']
    with pytest.raises(ValueError, match='Duplicate'):
        validate_handbook(bad)


def test_reject_cross_split_family_leakage():
    handbook = json.loads((ROOT / 'data/knowledge/handbook.json').read_text())
    cases = json.loads((ROOT / 'data/evaluation/assistant_cases.json').read_text())
    split = json.loads((ROOT / 'data/evaluation/assistant_split.json').read_text())
    dev = next(c for c in cases['cases'] if c['split'] == 'dev')
    test = next(c for c in cases['cases'] if c['split'] == 'test')
    test['scenario_family'] = dev['scenario_family']
    with pytest.raises(ValueError, match='family'):
        validate_cases(cases, split, handbook)
