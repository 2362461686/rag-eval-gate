import json

import pytest

import ragate.gate as gate_module
from ragate.gate import compare, main

QUERIES = [f"q{i}" for i in range(40)]


def _flat(value):
    return {query: value for query in QUERIES}


def _run(value):
    return {"ndcg@10": _flat(value), "recall@10": _flat(value), "mrr@10": _flat(value)}


def _by_metric(verdicts):
    return {verdict.metric: verdict for verdict in verdicts}


def test_identical_runs_pass_with_p_of_one():
    verdicts = compare(_run(0.5), _run(0.5), n_resamples=300)
    assert all(not v.regressed and v.delta == 0.0 and v.p_value == 1.0 for v in verdicts)


def test_a_large_uniform_drop_is_a_regression():
    verdicts = compare(_run(0.8), _run(0.5), n_resamples=300)
    assert all(v.regressed for v in verdicts)


def test_verdict_reports_means_delta_and_interval():
    verdict = compare(_run(0.75), _run(0.5), n_resamples=300)[0]
    assert (verdict.baseline, verdict.candidate, verdict.delta) == (0.75, 0.5, -0.25)
    assert (verdict.ci_low, verdict.ci_high) == (-0.25, -0.25)
    assert verdict.metric == "mrr@10"


def test_an_improvement_is_not_a_regression():
    assert not any(v.regressed for v in compare(_run(0.5), _run(0.8), n_resamples=300))


def test_a_significant_drop_below_the_margin_passes():
    verdicts = compare(_run(0.5), _run(0.495), margin=0.01, n_resamples=300)
    assert all(v.p_adjusted < 0.05 and not v.regressed for v in verdicts)


def test_a_drop_exactly_at_the_margin_fails():
    assert all(v.regressed for v in compare(_run(0.5), _run(0.25), margin=0.25, n_resamples=300))


def test_a_drop_just_inside_the_margin_passes():
    assert not any(v.regressed for v in compare(_run(0.5), _run(0.25), margin=0.26, n_resamples=300))


def test_alpha_is_a_strict_bound():
    observed = compare(_run(0.8), _run(0.5), n_resamples=300)[0].p_adjusted
    assert not any(v.regressed for v in compare(_run(0.8), _run(0.5), alpha=observed, n_resamples=300))
    assert all(v.regressed for v in compare(_run(0.8), _run(0.5), alpha=observed * 1.01, n_resamples=300))


def test_noise_around_zero_does_not_fail():
    base = {q: 0.5 for q in QUERIES}
    noisy = {q: 0.5 + (0.2 if i % 2 else -0.2) for i, q in enumerate(QUERIES)}
    verdicts = compare({"ndcg@10": base}, {"ndcg@10": noisy}, n_resamples=500)
    assert not verdicts[0].regressed


def test_p_values_are_adjusted_across_metrics(monkeypatch):
    returned = iter([0.02, 0.5, 0.5])
    monkeypatch.setattr(gate_module, "sign_flip_p", lambda *args: next(returned))
    verdicts = _by_metric(compare(_run(0.8), _run(0.5), n_resamples=100))
    assert verdicts["mrr@10"].p_value == 0.02
    assert verdicts["mrr@10"].p_adjusted == pytest.approx(0.06)
    assert not verdicts["mrr@10"].regressed


def test_gate_defaults_are_a_one_point_margin_at_five_percent(monkeypatch):
    monkeypatch.setattr(gate_module, "sign_flip_p", lambda *args: 0.0499)
    single = {"ndcg@10": _flat(0.5)}
    assert compare(single, {"ndcg@10": _flat(0.49)}, n_resamples=50)[0].regressed
    assert not compare(single, {"ndcg@10": _flat(0.495)}, n_resamples=50)[0].regressed
    monkeypatch.setattr(gate_module, "sign_flip_p", lambda *args: 0.0501)
    assert not compare(single, {"ndcg@10": _flat(0.4)}, n_resamples=50)[0].regressed


def test_results_are_reproducible_for_a_seed():
    assert compare(_run(0.8), _run(0.7), n_resamples=200, seed=9) == compare(_run(0.8), _run(0.7), n_resamples=200, seed=9)


def test_different_metrics_are_rejected():
    with pytest.raises(ValueError, match="same metrics"):
        compare({"a": _flat(0.5)}, {"b": _flat(0.5)})


def test_empty_runs_are_rejected():
    with pytest.raises(ValueError, match="same metrics"):
        compare({}, {})


def test_different_query_sets_are_rejected():
    fewer = {q: 0.5 for q in QUERIES[:-1]}
    with pytest.raises(ValueError, match="different queries"):
        compare({"ndcg@10": _flat(0.5)}, {"ndcg@10": fewer})


def _write(path, metrics):
    path.write_text(json.dumps({"metrics": metrics}), encoding="utf-8")
    return str(path)


def test_cli_exits_zero_when_clean(tmp_path, capsys):
    base = _write(tmp_path / "b.json", _run(0.5))
    assert main([base, base]) == 0
    assert capsys.readouterr().out.count("ok ") == 3


def test_cli_exits_one_on_regression(tmp_path, capsys):
    base, cand = _write(tmp_path / "b.json", _run(0.8)), _write(tmp_path / "c.json", _run(0.5))
    assert main([base, cand]) == 1
    assert capsys.readouterr().out.count("REGRESSED") == 3


def test_cli_passes_margin_and_alpha_through(tmp_path):
    base, cand = _write(tmp_path / "b.json", _run(0.8)), _write(tmp_path / "c.json", _run(0.5))
    assert main([base, cand, "--margin", "0.5"]) == 0
    assert main([base, cand, "--alpha", "0.00001"]) == 0


def test_cli_exits_two_on_invalid_input(tmp_path, capsys):
    base = _write(tmp_path / "b.json", _run(0.5))
    missing = tmp_path / "missing.json"
    malformed = tmp_path / "bad.json"
    malformed.write_text("{}", encoding="utf-8")
    other = _write(tmp_path / "other.json", {"ndcg@10": _flat(0.5)})
    assert main([base, str(missing)]) == 2
    assert main([base, str(malformed)]) == 2
    assert main([base, other]) == 2
    assert "invalid input" in capsys.readouterr().err
