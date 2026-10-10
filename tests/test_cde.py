"""CPU contract tests; these do not certify real GPU pooling or detector AUROC."""
import copy
import hashlib
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from reppl2.config_loader import load_config
from reppl2.cache import content_hash
from reppl2.semantic_views import (public_question, canonical_answer, validate_annotation,
                                  edited_question, choose_competitor, split_options)
from reppl2.semantic_readout import SemanticReadout
from reppl2.methods.reppl_e import (fit_calibrator, predict_calibrator, grouped_oof,
                                  uncertainty_features)
from reppl2 import cde
from reppl2.logging_utils import save_json, load_json


@pytest.fixture
def cfg():
    return load_config(Path(__file__).parents[1] / 'config/cde.yaml')


@pytest.fixture
def ex():
    return {'sample_id': 's1', 'task': 'open_qa', 'question': 'Where was Ada born?',
            'gold_answers': ['SECRET_GOLD'], 'label': 1,
            'meta': {'dataset': 'popqa', 'subj_id': 'ada'}}


def proposal():
    return {'units': [{'unit_id': 'u1', 'role': 'entity', 'text': 'Ada'}],
            'views': [{'question_stem': 'What is the birthplace of Ada?',
                       'units': [{'unit_id': 'u1', 'role': 'entity', 'text': 'Ada'}]}]}


def prepared(ex, cfg):
    ann = validate_annotation(public_question(ex), proposal(), cfg['cde'])
    ann['views'][1]['validation'] = 'model_checked_not_human_gold'
    return {'sample_id': ex['sample_id'], 'status': 'ok', 'annotation': ann,
            'question_hash': content_hash(public_question(ex)), 'human_reviewed': False}


def output(text, k=0):
    return dict(sample_id='s1', k=k, token_ids=[10, 11], text=text,
                token_logprobs=[-.2, -.4], finish_reason='stop')


def gen(identical=False):
    return {'sample_id': 's1', 'greedy': output('London'), 'prompt_text': 'test',
            'prompt_token_ids': [1, 2],
            'samples': [output('London'), output('London' if identical else 'Paris', 1)]}


class Tokenizer:
    eos_token = None
    eos_token_id = 0
    additional_special_tokens_ids = []
    def __call__(self, text, **kwargs):
        return {'input_ids': list(text.encode())}
    def apply_chat_template(self, messages, **kwargs):
        return '\n'.join(m['content'] for m in messages)


class Native:
    name = 'vllm'
    def __init__(self):
        self.calls = []
        self.closed = self.released = False
    def replay_last_hidden(self, context, output):
        self.calls.append((context, output))
        digest = hashlib.sha256(bytes(context + output)).digest()
        h = np.frombuffer(digest, np.uint8).astype(float) - 127.
        return {'hidden': h[None, :]}
    def release_for_replay(self):
        self.released = True
    def close(self):
        self.closed = True


def test_readout_suffix_cache_and_native_only(cfg):
    b = Native()
    read = SemanticReadout(b, Tokenizer(), cfg['cde'])
    a = read.observe('Q', 'A')
    assert np.linalg.norm(a) == pytest.approx(1)
    assert np.array_equal(a, read.observe('Q', 'A'))
    assert read.cost()['replay_calls'] == 1
    assert read.cost()['cache_hits'] == 1
    context, last = b.calls[0]
    assert len(last) == 1
    assert bytes(context + last).decode() == cfg['cde']['readout_template'].format(question='Q', candidate='A')
    b.name = 'transformers'
    with pytest.raises(ValueError, match='no Transformers'):
        SemanticReadout(b, Tokenizer(), cfg['cde'])


def test_mc_ten_options_resolve_full_semantics():
    ex = {'task': 'mcq', 'question': 'Select.\n' + '\n'.join(f'{c}. text {c}' for c in 'ABCDEFGHIJ'),
          'meta': {'n_options': 10}}
    stem, options, block = split_options(ex)
    assert stem == 'Select.'
    assert canonical_answer(output('Reasoning\nFinal answer: J'), options)['text'] == 'text J'
    assert choose_competitor({'id': 'option-9'}, [], options, 42)['id'] != 'option-9'
    assert block.startswith('A.')
    with pytest.raises(ValueError):
        canonical_answer(output('I think J might be correct'), options)


def test_annotation_blindness_and_checked_views(cfg, ex):
    seen = []
    def request(prompt, data):
        seen.append(data)
        return (proposal() if 'max_units' in data else
                {'equivalent': True, 'units_correspond': True}), {'input_tokens': 1}
    p = cde.prepare_one(ex, request, cfg['cde'])
    assert p['annotation']['views'][1]['validation'] == 'model_checked_not_human_gold'
    assert 'SECRET_GOLD' not in str(seen)
    view = p['annotation']['views'][0]
    assert edited_question(view, view['units'][0], 'an unspecified entity', '') == 'Where was an unspecified entity born?'
    assert p['human_reviewed'] is False


def test_invalid_views_and_whole_question_mask(cfg, ex):
    q = proposal()
    q['views'][0]['units'][0]['role'] = 'condition'
    assert validate_annotation(public_question(ex), q, cfg['cde'])['rejected_views']
    q = proposal()
    q['views'][0]['question_stem'] += ' in 1900'
    assert validate_annotation(public_question(ex), q, cfg['cde'])['rejected_views']
    q = proposal()
    q['units'][0]['text'] = ex['question']
    with pytest.raises(ValueError, match='entire question'):
        validate_annotation(public_question(ex), q, cfg['cde'])


def test_c_missing_competitor_d_survives_and_gold_has_no_effect(cfg, ex):
    p = prepared(ex, cfg)
    read = SemanticReadout(Native(), Tokenizer(), cfg['cde'])
    row = cde.detect_one_cde(ex, gen(True), p, read, cfg['cde'], {0})
    assert row['methods']['reppl-c']['risk'] is None
    assert row['methods']['reppl-d']['validity'] == 'ok'
    assert row['methods']['outer-perplexity']['risk'] == pytest.approx(.3)
    assert row['edits'] and row['propagation']['d']['u']
    changed = {**ex, 'gold_answers': ['DIFFERENT'], 'label': 0}
    reread = SemanticReadout(Native(), Tokenizer(), cfg['cde'])
    row2 = cde.detect_one_cde(changed, gen(True), p, reread, cfg['cde'], {0})
    assert row['methods'] == row2['methods']
    assert read.backend.calls == reread.backend.calls


def test_c_contribution_and_truncated_sample_coverage(cfg, ex):
    g = gen()
    g['samples'].append({**output('broken'), 'finish_reason': 'length'})
    row = cde.detect_one_cde(ex, g, prepared(ex, cfg),
                           SemanticReadout(Native(), Tokenizer(), cfg['cde']), cfg['cde'], {0})
    assert row['methods']['reppl-c']['validity'] == 'ok'
    assert row['c_sampling_coverage']['n_valid'] == 2
    assert len(row['propagation']['c']['a']) == 2
    assert row['methods']['reppl-c']['risk'] == pytest.approx(
        (row['propagation']['c']['inner'] + cfg['cde']['epsilon']) * .3)


def test_e_feature_allowlist_missing_and_positive_fit():
    d = {'validity': 'ok', 'components': [.1, .3], 'outer': 999., 'judge': 1.}
    assert np.allclose(uncertainty_features(None, d, 'd'), [.2, .3, .2])
    assert uncertainty_features(None, d, 'cd') is None
    X = np.array([[.1, .2], [.2, .1], [.8, .9], [.9, .8]])
    model = fit_calibrator(X, [0, 0, 1, 1], .1)
    inner, prob, contributions = predict_calibrator(model, X)
    assert np.all(np.asarray(model['weights']) >= 0)
    assert np.allclose(inner, contributions.sum(1))
    assert prob[-1] > prob[0]
    bad = {**model, 'weights': [float('nan'), 1.]}
    with pytest.raises(ValueError):
        predict_calibrator(bad, X)


def test_e_nested_group_exclusion_and_train_only_scales():
    rng = np.random.default_rng(3)
    y = np.tile([0, 1], 30)
    X = rng.uniform(.1, 1., (60, 3)) + y[:, None]
    groups = np.repeat([f'g{i}' for i in range(30)], 2)
    ids = [f's{i}' for i in range(60)]
    result = grouped_oof(X, y, groups, ids, 'd',
                        {'outer_folds': 3, 'inner_folds': 2, 'seed': 42,
                         'regularizations': [.01, .1], 'max_iter': 2000})
    assert len(result['predictions']) == len(ids)
    assert len({r['sample_id'] for r in result['predictions']}) == len(ids)
    for model in result['fold_models']:
        assert not set(model['train_groups']) & set(model['test_groups'])
        train = [ids.index(s) for s in model['train_ids']]
        assert np.allclose(model['scale'], np.sqrt((X[train] ** 2).mean(0)))


def test_native_model_impl_reaches_both_runners(monkeypatch):
    from reppl2.backends.vllm_backend import VLLMBackend
    module = types.ModuleType('vllm.config')
    module.PoolerConfig = lambda **kw: kw
    monkeypatch.setitem(sys.modules, 'vllm.config', module)
    b = object.__new__(VLLMBackend)
    b.cfg = {'model_impl': 'vllm'}
    b.model_path = '/fake'
    b._gen_llm = b._pool_llm = None
    seen = []
    b._load_with_retry = lambda kwargs, what: seen.append(kwargs) or object()
    b._gen_engine(); b._pool_engine()
    assert [v['model_impl'] for v in seen] == ['vllm', 'vllm']


def test_stage_resume_invalid_outer_and_cache_invalidation(tmp_path, cfg, ex, monkeypatch):
    out = tmp_path / 'result'
    out.mkdir()
    native = Native()
    monkeypatch.setattr(cde, '_backend', lambda *a: (native, Tokenizer()))
    examples = {'s1': ex, 's2': {**ex, 'sample_id': 's2'}}
    g2 = {**gen(), 'sample_id': 's2'}
    save_json(out / 'cde_prepared.json', {'identity': 'i', 'per_sample': [prepared(ex, cfg),
              {'sample_id': 's2', 'status': 'invalid', 'reason': 'bad view'}]})
    cde.detect_stage(cfg, logging.getLogger('test'), out, 'i', examples, {}, [gen(), g2])
    det = cde.verified_detection(out, 'i')
    assert native.released and native.closed
    assert det['per_sample'][1]['methods']['outer-perplexity']['risk'] == pytest.approx(.3)
    assert det['per_sample'][1]['methods']['reppl-d']['risk'] is None
    old_calls = len(native.calls)
    cde.detect_stage(cfg, logging.getLogger('test'), out, 'i', examples, {}, [gen(), g2])
    assert len(native.calls) == old_calls
    cde.evaluate_stage(cfg, out, 'i', {'s1': 0, 's2': 1},
                       {'label_source': 'em_gold', 'label_hash': 'test'}, examples)
    ev = load_json(out / 'cde_evaluation.json')
    assert ev['per_method']['outer-perplexity']['n_used'] == 2
    assert ev['per_method']['reppl-d']['n_invalid_scores'] == 1
    p = load_json(out / 'cde_prepared.json')
    p['per_sample'][0]['human_reviewed'] = True
    save_json(out / 'cde_prepared.json', p)
    with pytest.raises(ValueError, match='stale'):
        cde.verified_detection(out, 'i')


def test_frozen_source_and_output_separation(tmp_path, cfg, ex, monkeypatch):
    src = tmp_path / 'source'; src.mkdir()
    save_json(src / 'dataset.json', {'dataset': 'popqa', 'examples': [ex]})
    save_json(src / 'generation.json', {'model': 'fake', 'model_path': '/fake', 'generations': [gen()]})
    cfg['cde']['source_run'] = str(src)
    cfg['cde']['run_id'] = 'new'
    _, _, _, _, _, source = cde.load_source(cfg)
    monkeypatch.setattr(cde, 'results_dir', lambda *a: src)
    with pytest.raises(ValueError, match='overwrite source'):
        cde._output(cfg, source)
    assert not (src / 'config_snapshot.json').exists()


def test_e_stage_artifact_and_stale_labels(tmp_path, cfg):
    cfg['cde']['e'].update(outer_folds=3, inner_folds=2, regularizations=[.1])
    cfg['cde']['bootstrap_repeats'] = 5
    save_json(tmp_path / 'cde_prepared.json', {'identity': 'i', 'per_sample': []})
    ident = content_hash({'identity': 'i', 'prepared_sha': cde._sha(tmp_path / 'cde_prepared.json')})
    examples, rows, labels = {}, [], {}
    for i in range(60):
        sid = f's{i}'
        labels[sid] = i % 2
        examples[sid] = {'question': sid, 'meta': {'dataset': 'popqa', 'subj_id': f'g{i // 2}'}}
        d = {'validity': 'ok', 'components': [.1 + .3 * (i % 2), .2 + .2 * (i % 2)]}
        rows.append({'sample_id': sid, 'propagation': {'d': d},
                     'methods': {'outer-perplexity': {'risk': .1 + .02 * (i % 5), 'validity': 'ok'}}})
    save_json(tmp_path / 'cde_detection.json', {'identity': ident, 'per_sample': rows})
    meta = {'label_source': 'em_gold', 'label_hash': 'test'}
    cde.train_e_stage(cfg, tmp_path, 'i', examples, labels, meta)
    report = load_json(tmp_path / 'cde_e_oof.json')
    assert len(report['predictions']) == 60
    assert all(len(p['feature_values']) == 3 and p['model_hash'] for p in report['predictions'])
    cde.evaluate_stage(cfg, tmp_path, 'i', labels, meta, examples)
    ev = load_json(tmp_path / 'cde_evaluation.json')
    assert ev['paired_outer']['reppl-e']['n_groups'] == 30
    assert ev['per_method']['reppl-e']['n_used'] == 60
    with pytest.raises(ValueError, match='different labels'):
        cde.evaluate_stage(cfg, tmp_path, 'i', labels, {**meta, 'label_hash': 'changed'}, examples)


def test_invalid_semantic_verifier_preserves_c_units(cfg, ex):
    def request(prompt, data):
        if 'max_units' in data:
            return proposal(), {}
        raise ValueError('invalid JSON')
    row = cde.prepare_one(ex, request, cfg['cde'])
    assert row['status'] == 'ok'
    assert len(row['annotation']['views']) == 1
    assert row['annotation']['rejected_views']


def test_judge_label_identity_and_nonbinary_rejected(tmp_path, cfg, ex):
    cfg['cde']['label_source'] = 'judge'
    save_json(tmp_path / 'judge.json', {'_config_hash_': 'same', 'per_sample': [
        {'sample_id': 's1', 'hard_verdict': .9}]})
    with pytest.raises(ValueError, match='binary'):
        cde.evaluation_labels(cfg, tmp_path, {'s1': ex}, {'_config_hash_': 'same'}, [gen()])
    with pytest.raises(ValueError, match='hashes'):
        cde.evaluation_labels(cfg, tmp_path, {'s1': ex}, {'_config_hash_': 'different'}, [gen()])
