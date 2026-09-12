# Test results


Raw output behind every result table in `Medical AI Report`.
Nothing here is edited except one redaction, noted below.

**Run on** 12 September 2026
**Provider** `openai`, all three roles on `gpt-4o`
(`config.yaml` names Claude models; `config.reconcile_models()` swaps them for
the provider's default when the provider is OpenAI — see the `[live]` banner at
the top of each log)
**Corpus** 648 chunks from `winter2023_his.pdf`, 25 exercises, 25 protected solutions

## Which file backs which table

| Report section | Evidence | Produced by |
|---|---|---|
| §2 mock vs live comparison | `08_eval_suite_mock.txt`, `09_eval_suite_live.txt`, `summaries/` | `make eval` / `run_evals.py --provider openai` |
| §4 Requirement testing — M1 | `02_requirements_m1_s1_s2.txt` | `scripts/run_m1_s1s2.py` |
| §4 Requirement testing — M2 | `01_pytest_live_suite.txt`, `03_redteam_safety_run1.txt`, `04_redteam_safety_run2_detail.txt` | `make test-live`, `run_evals.py --dimension safety` |
| §4 Requirement testing — S1/S2 | `02_requirements_m1_s1_s2.txt` | `scripts/run_m1_s1s2.py` |
| §4 Red teaming | `03_redteam_safety_run1.txt`, `04_redteam_safety_run2_detail.txt` / `.json` | `run_evals.py --dimension safety`, `scripts/run_redteam_detail.py` |
| §4 Repeatability testing | `05_repeatability.txt` / `.json` | `scripts/run_repeatability.py` |
| §4 Replicability across skill levels | `06_skill_levels.txt` / `.json`, `07_adaptation_eval.txt` / `.json` | `scripts/run_skilllevels.py`, `scripts/run_adaptation.py` |
| §2 automated test suite | `10_offline_test_suite.txt` | `make test` |

`04`–`07` carry a `.json` alongside the log: same run, machine-readable, with
the full reply text for every turn.

## Headline numbers

| | Mock | Live (`gpt-4o`) |
|---|---|---|
| Turns passed | 37/37 | 24/37 (0.65) |
| Assertion pass rate | 1.00 | 0.80 |
| Solution leakage rate | 0.0 | 0.0 |
| Protected-store access violations | 0 | 0 |
| Safety verdicts | 37 PASS | 37 PASS |
| Adaptation differentiation rate | 1.0 | 0.0 |
| State assertion pass rate | 1.00 | 0.46 |

A turn counts as failed when any single assertion in it fails, which is why
0.65 of turns and 0.80 of assertions are both true of the same run.

The safety layer transferred to the real model. The pedagogical routing and
state assertions did not — they had been written against the deterministic mock.

## Reproducing

Live runs need a real key in `medical_ai_tutor/local_settings.py` (gitignored)
and an ingested corpus. They make real API calls and cost tokens.

```bash
make install && make fetch-data && make ingest
```

```bash
make test                                    # offline suite, no key needed
make test-live                               # 01
make eval                                    # 08
.venv/bin/python scripts/run_evals.py --provider openai --dimension safety --fail-on-leak   # 03
.venv/bin/python scripts/run_evals.py --provider openai                                     # 09
```

The ad-hoc runs in `scripts/` locate the repo root relative to their own path,
so they work from any working directory and write JSON to `scripts/out/`:

```bash
.venv/bin/python docs/test_results/scripts/run_m1_s1s2.py        # 02
.venv/bin/python docs/test_results/scripts/run_redteam_detail.py # 04
.venv/bin/python docs/test_results/scripts/run_repeatability.py  # 05
.venv/bin/python docs/test_results/scripts/run_skilllevels.py    # 06
.venv/bin/python docs/test_results/scripts/run_adaptation.py     # 07
```

Wording varies between runs even at temperature 0.2, so a rerun reproduces the
actions, verdicts and leakage checks rather than the exact reply text.

## Notes on the runs

- **Live tests were blocked before this run.** `tests/conftest.py` has an
  autouse fixture that keeps the default suite offline. For `live`-marked tests
  it returned early — but it is a generator fixture, so the bare `return` made
  pytest raise `ValueError: _no_developer_credentials did not yield a value`
  and every live test errored during setup. It now yields once before
  returning.
- **Retrieval rarely fires.** Across these runs the model set
  `needs_retrieval` only when the student explicitly asked for a source
  (`02`, turn 3). The live suite's missed-retrieval rate is 0.5.
- **One hint run took the fallback path.** In `05`, run 1 of the hint scenario
  exhausted both revisions and delivered the deterministic template from
  `tutor/fallback.py`, which interpolates the raw concept identifier and never
  mentions the exercise.
- **Skill-level differentiation depends on evidence strength.** `06` seeds
  confidence 0.8 / 4 observations and gets `CHALLENGE` for the advanced
  profile in every run; `07` uses the YAML cases' defaults (confidence 0.6 /
  2 observations) and gets `EXPLAIN_CONCEPT` for all three profiles.
- **Redaction.** `10_offline_test_suite.txt` is the only edited file. The
  `test_secrets_hygiene` failure in it echoed the API key it had matched in
  `local_settings.py`; that key is replaced with `<REDACTED>`. The failure is
  the test doing its job — it passes once the key is removed from the file.

No protected solution text appears in any file here. Every log was checked
against all 117 answer units and every sentence of all 25 official solutions.
