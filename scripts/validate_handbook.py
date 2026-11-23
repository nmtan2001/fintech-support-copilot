"""Validate the authored handbook and frozen assistant evaluation without mutation."""
import json
from collections import Counter
from datetime import date
from pathlib import Path

TOPICS = {'card_payments', 'card_status', 'card_delivery', 'cash_withdrawals', 'transfers',
          'recipient_details', 'refunds', 'disputes', 'topups', 'fees_fx', 'limits', 'account_security'}


def validate_handbook(document):
    if document.get('synthetic_only') is not True or not document.get('version') or not document.get('brand'):
        raise ValueError('Handbook must identify its fictional brand, version and synthetic scope')
    articles = document.get('articles', [])
    ids = [a.get('id') for a in articles]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate article id')
    if len(articles) != 48 or Counter(a.get('topic') for a in articles) != Counter({t: 4 for t in TOPICS}):
        raise ValueError('Expected 48 articles with four in each of twelve topics')
    for article in articles:
        for field in ('id', 'title', 'topic', 'version', 'effective_date', 'content', 'source'):
            if not isinstance(article.get(field), str) or not article[field].strip():
                raise ValueError('Missing article field: '+field)
        date.fromisoformat(article['effective_date'])
        if article['version'] != document['version'] or len(article['content']) < 150:
            raise ValueError('Invalid article version or empty policy content')
        related = article.get('related_ids')
        if not isinstance(related, list) or any(r not in ids or r == article['id'] for r in related):
            raise ValueError('Broken related article reference')
    return document


def validate_cases(document, split, handbook):
    cases = document.get('cases', [])
    expected = {'single': 80, 'cross_article': 40, 'multi_turn': 40, 'scope': 40, 'adversarial': 40}
    if Counter(c.get('category') for c in cases) != Counter(expected):
        raise ValueError('Invalid scenario category counts')
    if len(split.get('dev', [])) != 80 or len(split.get('test', [])) != 160:
        raise ValueError('Expected frozen 80/160 split')
    ids = [c['id'] for c in cases]
    if len(set(ids)) != 240 or set(split['dev']) & set(split['test']) or set(split['dev']+split['test']) != set(ids):
        raise ValueError('Duplicate, missing or overlapping case ids')
    families = {}
    article_ids = {a['id'] for a in handbook['articles']}
    for case in cases:
        partition = case['split']
        if partition not in {'dev', 'test'} or case['id'] not in split[partition]:
            raise ValueError('Case partition does not match frozen split')
        family = case['scenario_family']
        if family in families and families[family] != partition:
            raise ValueError('Scenario family leaked across partitions')
        families[family] = partition
        if case['expected_action'] not in {'answered', 'clarification_required', 'handoff'}:
            raise ValueError('Invalid expected action')
        if not case['required_facts'] or not case['messages']:
            raise ValueError('Missing gold facts or messages')
        if any(i not in article_ids for i in case['evidence_ids']):
            raise ValueError('Unknown evidence reference')
        if case['expected_action'] == 'answered' and not case['evidence_ids']:
            raise ValueError('Answerable cases require gold evidence')
        if any(m.get('role') not in {'user', 'assistant'} or not m.get('content', '').strip()
               for m in case['messages']):
            raise ValueError('Invalid conversation messages')
    return document


def main():
    root = Path(__file__).resolve().parents[1]
    handbook = validate_handbook(json.loads((root/'data/knowledge/handbook.json').read_text()))
    validate_cases(json.loads((root/'data/evaluation/assistant_cases.json').read_text()),
                   json.loads((root/'data/evaluation/assistant_split.json').read_text()), handbook)
    print('Validated 48 fictional articles, 240 authored scenarios and frozen 80/160 family-disjoint split.')


if __name__ == '__main__':
    main()
