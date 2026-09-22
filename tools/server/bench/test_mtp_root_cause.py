"""Deterministic parser tests for the bounded dense-MTP root-cause proof."""

from __future__ import annotations

import importlib.util
import pathlib
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("validate-mtp-root-cause.py")
SPEC = importlib.util.spec_from_file_location("validate_mtp_root_cause", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MTPRootCauseTest(unittest.TestCase):
    def test_dense_boundary_parser_preserves_extent_and_nonfinite_counts(self) -> None:
        lines = [
            "DENSE_DIAG name=Ksource-3 type=turbo4 elements=4194304 finite=-1 nan=-1 inf=-1 shape=[256,4,4096,1]",
            "DENSE_DIAG name=kqv_out-3 type=f32 elements=18432 finite=0 nan=18432 pos_inf=0 neg_inf=0 min=0 max=0",
        ]
        records = MODULE.dense_records(lines)
        self.assertEqual([256, 4, 4096, 1], MODULE.first_record(records, "Ksource-3")["shape"])
        self.assertEqual(18432, MODULE.first_nonfinite(records, "kqv_out-3")["nan"])

    def test_mtp_event_parser_identifies_bad_and_good_verification(self) -> None:
        bad = (
            'prefix MTP_STATE_DIAGNOSTIC {"proposal_positions":[31,32],'
            '"logits_finite_count":[0,0,0],"logits_nan_count":[4,4,4]}'
        )
        good = (
            'prefix MTP_STATE_DIAGNOSTIC {"proposal_positions":[31,32],'
            '"logits_finite_count":[8,8,8],"logits_nan_count":[0,0,0]}'
        )
        events = MODULE.mtp_events([bad, good])
        self.assertIsNotNone(MODULE.verification_event(events, True))
        self.assertIsNotNone(MODULE.verification_event(events, False))


if __name__ == "__main__":
    unittest.main()
