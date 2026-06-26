from argparse import Namespace

from torch_osqp_stability import environment_manifest, source_tree_fingerprint


def test_source_tree_fingerprint_is_deterministic_and_nonempty():
    first = source_tree_fingerprint()
    second = source_tree_fingerprint()
    assert first == second
    digest, file_count = first
    assert len(digest) == 64
    assert file_count > 20


def test_manifest_records_exact_source_provenance():
    args = Namespace(
        device="cpu",
        dtype="float64",
        seeds=1,
        stress_seeds=0,
        time_limit_seconds=7200.0,
    )
    manifest = environment_manifest(args)
    assert len(manifest["source_tree_sha256"]) == 64
    assert manifest["source_file_count"] > 20
    assert isinstance(manifest["git_status_entries"], (list, type(None)))
