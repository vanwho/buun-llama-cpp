"""Deterministic contract tests for the bounded MTP diagnostic harness."""

from __future__ import annotations

import pathlib
import runpy
import unittest

from mtp_diagnostic import (
    RUNG_SPECS,
    Rung,
    build_server_argv,
    cold_sequence_prompt,
    command_contract,
    parse_prometheus,
    effective_context,
    validate_prompt_tokens,
    request_fields,
    validate_request_record,
)


class MTPDiagnosticTest(unittest.TestCase):
    def test_exact_rung_commands_are_bounded_and_turbo4(self) -> None:
        binary = pathlib.Path("/opt/llama.cpp/build-cuda/bin/llama-server")
        model = pathlib.Path("/srv/ai/models/text/current.gguf")
        for rung in RUNG_SPECS:
            context = effective_context(4096, rung)
            argv = build_server_argv(binary, model, 18080, rung, context=context)
            self.assertEqual([], command_contract(argv, rung, context=context, batch=128, ubatch=128))
            self.assertEqual(str(context), argv[argv.index("-c") + 1])
            self.assertEqual("128", argv[argv.index("-b") + 1])
            self.assertEqual("128", argv[argv.index("-ub") + 1])
            self.assertEqual("turbo4", argv[argv.index("-ctk") + 1])
            self.assertEqual("turbo4", argv[argv.index("-ctv") + 1])
            self.assertEqual("auto", argv[argv.index("--kv-safety-headroom") + 1])

    def test_actual_prompt_token_contract_limit_and_one_over(self) -> None:
        self.assertEqual([], validate_prompt_tokens(16368, 16384, 16))
        self.assertIn("prompt_tokens_exceed_global_max", validate_prompt_tokens(16385, 16384, 0))
        self.assertIn("prompt_tokens_exceed_context_reserve", validate_prompt_tokens(16369, 16384, 16))
        self.assertIn("prompt_tokens_exceed_context_reserve", validate_prompt_tokens(8180, 8192, 16))

    def test_safety_headroom_is_an_identity_contract(self) -> None:
        rung = RUNG_SPECS[3]
        argv = build_server_argv(pathlib.Path("server"), pathlib.Path("model"), 18080,
                                 rung, context=8192, batch=128, ubatch=64)
        self.assertEqual([], command_contract(argv, rung, context=8192, batch=128, ubatch=64))
        argv[argv.index("--kv-safety-headroom") + 1] = "0"
        self.assertIn("--kv-safety-headroom=auto required",
                      command_contract(argv, rung, context=8192, batch=128, ubatch=64))
        oversized = Rung("oversized", "selective", True, 193, 49153, "over budget")
        oversized_argv = build_server_argv(pathlib.Path("server"), pathlib.Path("model"),
                                           18080, oversized, context=4096)
        self.assertIn("hot_page_budget_exceeds_49152",
                      command_contract(oversized_argv, oversized,
                                       context=4096, batch=128, ubatch=128))

    def test_only_cold_rung_raises_context_and_hot_budget_is_bounded(self) -> None:
        self.assertEqual(4096, effective_context(4096, RUNG_SPECS[2]))
        self.assertEqual(8192, effective_context(4096, RUNG_SPECS[3]))
        self.assertLessEqual(RUNG_SPECS[2].hot_pages * 256, 49152)
        self.assertLessEqual(RUNG_SPECS[3].hot_pages * 256, 49152)

    def test_cold_documents_are_three_prefix_preserving_real_queries(self) -> None:
        prompts = [cold_sequence_prompt(i) for i in (1, 2, 3)]
        self.assertTrue(prompts[1].startswith(prompts[0]))
        self.assertTrue(prompts[2].startswith(prompts[1]))
        self.assertIn("DOCUMENT_A", prompts[0])
        self.assertIn("AZURE-731", prompts[0])
        self.assertIn("DOCUMENT_B", prompts[1])
        self.assertIn("BRASS-284", prompts[1])
        self.assertIn("DOCUMENT_A", prompts[2])
        self.assertGreater(len(prompts[1]), 10000)

    def test_post_repair_validator_requires_cold_movement_chain(self) -> None:
        validate = runpy.run_path(str(pathlib.Path(__file__).with_name(
            "validate-mtp-post-repair.py")))
        proof = {
            "page_cold_before_request": True, "h2d_completed": True,
            "mapping_published": True, "target_consumed": True, "draft_consumed": True,
            "cold_logical_page_id": 4, "selected_logical_page_id": 4,
            "cold_generation": 9, "selected_generation": 9,
            "cold_content_version": 12, "selected_content_version": 12,
            "event_order": {"h2d_completed": 1, "mapping_published": 2,
                             "graph_consumed": 3},
        }
        prompts = ["A", "AB", "ABC"]
        records = [{"cold_sequence": stage, "request": {"prompt": prompt}}
                   for stage, prompt in zip(("ingest_document_a", "append_document_b_query_b",
                                             "query_document_a_again"), prompts)]
        records[2]["promotion_proof"] = proof
        self.assertEqual([], validate["_cold_sequence_errors"](records))
        records[2]["promotion_proof"]["h2d_completed"] = None
        self.assertIn("cold promotion movement/consumption proof missing",
                      validate["_cold_sequence_errors"](records))

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
            {"request": {"n_predict": 8}, "request_fields": fields,
             "request_contract": {"configured_context_tokens": 4096,
                                  "rendered_prompt_tokens": 32}}, rung))

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
                  "request_contract": {"configured_context_tokens": 4096,
                                       "rendered_prompt_tokens": 32},
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

    def test_dense_control_records_explicit_non_mtp_fields(self) -> None:
        fields = request_fields(
            {"predicted_tokens": 10},
            {"predicted_tokens": 12}, {}, {}, {}, mtp=False)
        self.assertEqual(0, fields["draft_n"])
        self.assertEqual(0, fields["draft_n_accepted"])
        self.assertEqual(0, fields["verification_steps"])
        self.assertEqual("dense", fields["pager_route"])
        self.assertEqual("not_applicable_dense", fields["page_table_epoch"])
        self.assertEqual("not_present", fields["mtp_placement"])


if __name__ == "__main__":
    unittest.main()
