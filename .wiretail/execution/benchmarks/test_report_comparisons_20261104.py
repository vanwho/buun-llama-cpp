"""Small synthetic checks for the offline comparison report importer."""
import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("report-comparisons-20261104.py")
SPEC = importlib.util.spec_from_file_location("comparison_report", SCRIPT)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def measured(rep, *, status="pass", http=200, complete=True, rate=10.0, tokens=40):
    return {
        "kind": "measured", "status": status, "http_status": http,
        "stream_complete": complete, "repetition": rep, "prompt_index": 0,
        "reasoning": "low", "decode_tok_s": rate,
        "completion_tokens": tokens, "actual_C": 126900 + rep,
        "actual_C_after": 126950 + rep,
        "nominal_target_C": 126976,
    }


class ComparisonImporterChecks(unittest.TestCase):
    def test_only_complete_valid_triplet_is_selected_and_actual_c_is_per_repeat(self):
        records = [measured(1), measured(2), measured(3)]
        triplet = REPORT.validated_triplet(records, 0, "low")
        self.assertEqual([r["repetition"] for r in triplet], [1, 2, 3])
        cell = REPORT.cell("fork-cpu-mtp-off", 126976, "low", 0, triplet, {}, {}, "test", False, 126976)
        self.assertEqual(cell["actual_C_repeats"], [126901, 126902, 126903])
        self.assertEqual(cell["nominal_target_C"], 126976)

    def test_incomplete_duplicate_http_fault_and_short_outputs_do_not_form_triplet(self):
        self.assertIsNone(REPORT.validated_triplet([measured(1), measured(1), measured(2)], 0, "low"))
        self.assertIsNone(REPORT.validated_triplet([measured(1), measured(2), measured(3, http=500)], 0, "low"))
        self.assertIsNone(REPORT.validated_triplet([measured(1), measured(2), measured(3, complete=False)], 0, "low"))
        self.assertIsNone(REPORT.validated_triplet([measured(1), measured(2), measured(3, tokens=3)], 0, "low"))
        self.assertIsNone(REPORT.validated_triplet([measured(1), measured(2), measured(3, rate=float("nan"))], 0, "low"))

    def test_partial_campaign_without_row_recovers_nominal_and_actual_separately(self):
        cfg = {"requested_target_tokens": 126976}
        actual, nominal = REPORT.campaign_fallback_geometry(
            [measured(1), measured(2), measured(3)], cfg, {}
        )
        self.assertEqual(nominal, 126976)
        self.assertEqual(actual, 126953)

    def test_unknown_actual_occupancy_is_not_replaced_by_nominal_target(self):
        actual, nominal = REPORT.campaign_fallback_geometry(
            [{"nominal_target_C": 126976}], {"requested_target_tokens": 126976}, {}
        )
        self.assertIsNone(actual)
        self.assertEqual(nominal, 126976)

    def test_quality_classifier_keeps_short_eos_and_reasoning_only(self):
        self.assertEqual(REPORT.response_quality_from_text("OK.")["classification"], "no_repeated_punctuation_failure")
        self.assertEqual(REPORT.response_quality_from_text("reasoning: explain the answer")["classification"], "no_repeated_punctuation_failure")
        self.assertEqual(REPORT.response_quality_from_text("/" * 400)["classification"], "repeated_punctuation_only")

    def test_one_bad_repetition_excludes_whole_triplet_without_copying_text(self):
        REPORT.RESPONSE_QUALITY_EXCLUSIONS.clear()
        REPORT._RECORD_CACHE.clear()
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td) / "fork-cpu-mtp-on" / "attempt-001-L32768" / "cell" / "C6144" / "campaign"
            directory.mkdir(parents=True)
            records=[]
            for rep in (1, 2, 3):
                raw=directory/f"response-{rep}.sse"
                text=("/"*400) if rep==2 else "Let me think carefully.\n42"
                raw.write_text("data: "+json.dumps({"choices":[{"delta":{"content":text}}]})+"\ndata: [DONE]\n",encoding="utf8")
                records.append({"kind":"measured","status":"pass","http_status":200,"stream_complete":True,"repetition":rep,"prompt_index":0,"reasoning":"low","decode_tok_s":10+rep,"completion_tokens":40,"actual_C_before":6144+rep,"actual_C_after":6200+rep,"mtp":{"acceptance_percent":5.0},"raw_path":str(raw),"raw_sha256":hashlib.sha256(raw.read_bytes()).hexdigest()})
            record_path=directory/"records.jsonl"
            record_path.write_text("".join(json.dumps(r)+"\n" for r in records),encoding="utf8")
            cells={}
            REPORT.add_raw_family(cells,"fork-cpu-mtp-on",[(6144,str(record_path),None,{},None,False,False,6144)],{},"fixture",True,{})
            self.assertEqual(cells,{})
            self.assertEqual(len(REPORT.RESPONSE_QUALITY_EXCLUSIONS),1)
            exclusion=REPORT.RESPONSE_QUALITY_EXCLUSIONS[0]
            self.assertEqual(exclusion["bad_repetitions"],[2])
            self.assertNotIn("text",json.dumps(exclusion))

    def test_missing_raw_response_is_unassessed_not_rejected(self):
        REPORT.RESPONSE_QUALITY_EXCLUSIONS.clear()
        REPORT.RESPONSE_QUALITY_UNASSESSED.clear()
        REPORT._RECORD_CACHE.clear()
        with tempfile.TemporaryDirectory() as td:
            directory=Path(td)/"fork-cpu-mtp-on"/"attempt-001-L32768"/"cell"/"C6144"/"campaign"
            directory.mkdir(parents=True); record_path=directory/"records.jsonl"
            records=[dict(measured(rep),mtp={"acceptance_percent":3.0},raw_path=str(directory/f"missing-{rep}.sse"),raw_sha256="0"*64) for rep in (1,2,3)]
            record_path.write_text("".join(json.dumps(r)+"\n" for r in records),encoding="utf8")
            cells={}
            REPORT.add_raw_family(cells,"fork-cpu-mtp-on",[(6144,str(record_path),None,{},None,False,False,6144)],{} ,"fixture",True,{})
            self.assertEqual(len(cells),1)
            self.assertEqual(REPORT.RESPONSE_QUALITY_EXCLUSIONS,[])
            self.assertEqual(REPORT.RESPONSE_QUALITY_UNASSESSED[0]["missing_raw_repetitions"],[1,2,3])

    def test_same_nominal_does_not_relabel_actual_occupancy(self):
        left = REPORT.cell("fork-cpu-mtp-on", 126976, "low", 0,
                           [measured(1), measured(2), measured(3)], {}, {}, "test", False, 126976)
        right_triplet = [dict(measured(i), actual_C=127000+i) for i in (1, 2, 3)]
        right = REPORT.cell("fork-cpu-mtp-on", 127003, "low", 0,
                            right_triplet, {}, {}, "test", False, 126976)
        self.assertEqual(left["nominal_target_C"], right["nominal_target_C"])
        self.assertNotEqual(left["actual_C"], right["actual_C"])


if __name__ == "__main__":
    unittest.main()
