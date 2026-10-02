"""The stored reference files themselves (#379): strict JSON, provenance
on every entry, and fixtures that the generator reproduces."""

import importlib.util
import json
from pathlib import Path

import pytest

from ._data import DATA, sources

REPO = Path(__file__).resolve().parents[3]
GENERATOR = REPO / "scripts" / "reference" / "make_fixtures.py"

EXPECTED_SOURCES = {
    "r_survival",
    "r_cmprsk",
    "r_timereg",
    "r_pec",
    "r_riskregression",
    "r_npsurv",
    "r_fitdistrplus",
    "r_frailty",
    "py_lifelines",
    "py_sksurv",
}


def _strict_load(path):
    def refuse(token):
        raise ValueError(
            "non-strict JSON constant {} in {}".format(token, path)
        )

    with open(path) as f:
        return json.load(f, parse_constant=refuse)


def test_every_reference_file_is_present():
    assert set(sources()) == EXPECTED_SOURCES


@pytest.mark.parametrize("source", sorted(EXPECTED_SOURCES))
def test_reference_file_is_strict_json_with_provenance(source):
    content = _strict_load(DATA / "{}.json".format(source))
    fixtures = _strict_load(DATA / "fixtures.json")
    assert content["generator"].startswith("scripts/reference/")
    assert content["references"]
    for ref_id, entry in content["references"].items():
        for key in ("fixture", "software", "version", "call", "settings"):
            assert entry[key], (source, ref_id, key)
        assert entry["fixture"] in fixtures, (source, ref_id)
        assert entry["values"], (source, ref_id)


@pytest.mark.skipif(
    not GENERATOR.exists(), reason="needs the repository's scripts folder"
)
def test_fixtures_are_reproduced_by_their_generator():
    spec = importlib.util.spec_from_file_location("make_fixtures", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rebuilt = json.loads(json.dumps(module.build(), allow_nan=False))
    assert rebuilt == _strict_load(DATA / "fixtures.json")
