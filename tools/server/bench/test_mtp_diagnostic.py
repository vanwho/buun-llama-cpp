"""Deterministic contract tests for the bounded MTP diagnostic harness."""

from __future__ import annotations

import pathlib
import unittest

from mtp_diagnostic import (
    RUNG_SPECS,
    build_server_argv,
    command_contract,
    parse_prometheus,
    request_fields,
    validate_request_record,
)


class MTPDiagnosticTest(unittest.TestCase):
    def test_exact_rung_commands_are_bounded_and_turbo4(self) -> None:
        binary = pathlib.Path("/opt/llama.cpp/build-cuda/bin/llama-server")
        model = pathlib.Path("/srv/ai/models/text/current.gguf")
        for rung in RUNG_SPECS:
            argv = build_server_argv(binary, model, 18080, rung)
            self.assertEqual([], command_contract(argv, rung, context=4096, batch=128, ubatch=128))
            self.assertEqual("4096", argv[argv.index("-c") + 1])
            self.assertEqual("128", argv[argv.index("-b") + 1])
            self.assertEqual("128", argv[argv.index("-ub") + 1])
            self.assertEqual("turbo4", argv[argv.index("-ctk") + 1])
            self.assertEqual("turbo4", argv[argv.index("-ctv") + 1])

    def test_prometheus_parser_keeps_request_counters_distinct(self) -> None:
        values = parse_prometheus(
            "llamacpp:kv_pager_predicted_tokens 8\n"
            "llamacpp:spec_decode_num_draft_tokens_total 6\n"
            "llamacpp:spec_decode_num_accepted_tokens_total 3\n"
            "llamacpp:spec_decode_num_drafts_total 2\n")
        self.assertEqual(8, values["predicted_tokens"])
        self.assertEqual(6, values["mtp_draft_tokens_total"])
        self.assertEqual(3, values["mtp_accepted_tokens_total"])
        self.assertEqual(2, values["mtp_verification_steps_total"])

    def test_missing_request_field_fails_instead_of_becoming_zero(self) -> None:
        rung = RUNG_SPECS[1]
        record = {
            "request": {"n_predict": 16},
            "request_fields": {
                "draft_n": 2, "draft_n_accepted": 1, "accepted_tokens": 1,
                "verification_steps": 1, "target_positions": [4],
                "draft_positions": [5], "rollback_count": 1,
                # rewind_count intentionally absent
                "pager_route": "dense", "page_table_epoch": 1,
                "mtp_placement": "gpu", "mtp_type_k": "turbo4",
                "mtp_type_v": "turbo4",
            },
        }
        self.assertIn("missing_rewind_count", validate_request_record(record, rung))

    def test_reference_and_cpu_routes_are_refused(self) -> None:
        rung = RUNG_SPECS[1]
        fields = {
            "draft_n": 2, "draft_n_accepted": 1, "accepted_tokens": 1,
            "verification_steps": 1, "target_positions": [4],
            "draft_positions": [5], "rollback_count": 1, "rewind_count": 0,
            "pager_route": "selected reference", "page_table_epoch": 1,
            "mtp_placement": "cpu", "mtp_type_k": "q4_0", "mtp_type_v": "q4_0",
        }
        errors = validate_request_record({"request": {"n_predict": 8},
                                          "request_fields": fields}, rung)
        self.assertIn("route_refused_or_unknown", errors)
        self.assertIn("mtp_not_gpu", errors)
        self.assertIn("mtp_type_k_not_turbo4", errors)

    def test_off_control_requires_explicit_not_present_mtp(self) -> None:
        rung = RUNG_SPECS[0]
        fields = {
            "draft_n": 0, "draft_n_accepted": 0, "accepted_tokens": 0,
            "verification_steps": 0, "target_positions": [4],
            "draft_positions": [5], "rollback_count": 0, "rewind_count": 0,
            "pager_route": "dense", "page_table_epoch": 1,
            "mtp_placement": "not_present", "mtp_type_k": "not_present",
            "mtp_type_v": "not_present",
        }
        self.assertEqual([], validate_request_record(
            {"request": {"n_predict": 8}, "request_fields": fields}, rung))

    def test_selected_resident_and_cold_contracts(self) -> None:
        resident = RUNG_SPECS[2]
        fields = {
            "draft_n": 2, "draft_n_accepted": 2, "accepted_tokens": 2,
            "verification_steps": 1, "target_positions": [4],
            "draft_positions": [5], "rollback_count": 0, "rewind_count": 0,
            "pager_route": "selected direct", "page_table_epoch": 2,
            "mtp_placement": "gpu", "mtp_type_k": "turbo4", "mtp_type_v": "turbo4",
        }
        record = {"request": {"n_predict": 8}, "request_fields": fields,
                  "pager_after": {"logical_pages": 2, "resident_pages": 2}}
        self.assertEqual([], validate_request_record(record, resident))
        cold = RUNG_SPECS[3]
        record["pager_before"] = {"h2d_useful_bytes": 100}
        record["pager_after"] = {"h2d_useful_bytes": 200}
        self.assertEqual([], validate_request_record(record, cold))

    def test_request_fields_preserve_absent_values(self) -> None:
        fields = request_fields({}, {}, {}, {"pager_metrics": {}}, {}, mtp=True)
        self.assertIsNone(fields["draft_n"])
        self.assertIsNone(fields["pager_route"])


if __name__ == "__main__":
    unittest.main()
