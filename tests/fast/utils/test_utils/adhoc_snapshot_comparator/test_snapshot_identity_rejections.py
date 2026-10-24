import pytest

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_parser_evidence import _actions_by_identity
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_variant_audit import _diagnostic_views


class TestSnapshotIdentityRejections:
    def test_two_physical_processes_cannot_share_one_canonical_owner(self) -> None:
        """A saved alias cannot claim an absent physical process shares the observed canonical owner."""
        row = {
            "variant_case": "variant",
            "canonical_case": "canonical",
            "process_identity_correspondences": [
                {"variant_process": "physical-a", "canonical_process": "actor"},
                {"variant_process": "physical-b", "canonical_process": "actor"},
            ],
            "transformed_before_process_correspondences": [],
            "transformed_before_differences": [],
        }

        with pytest.raises(AssertionError, match="Ambiguous process aliases"):
            _diagnostic_views(rows=[row], original={}, actual={"variant": {"physical-a": {}}})

    def test_duplicate_parser_identity_cannot_hide_different_action_behavior(self) -> None:
        """Two actions with identical identity must fail instead of overwriting a changed default."""
        actions = [
            {"dest": "enabled", "option_strings": ["--enabled"], "default": False},
            {"dest": "enabled", "option_strings": ["--enabled"], "default": True},
        ]

        with pytest.raises(AssertionError, match="Duplicate parser action identity"):
            _actions_by_identity(actions=actions)
