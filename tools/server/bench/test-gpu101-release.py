#!/usr/bin/env python3
"""Regression coverage for dependency-driven release scheduling."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "gpu101_release", Path(__file__).with_name("validate-gpu101-release.py"))
assert SPEC is not None and SPEC.loader is not None
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def chain(*statuses: str) -> list[dict]:
    names = ["measure", "repair", "retest", "review", "scale"]
    return [{"id": name, "status": status,
             "depends_on": [] if i == 0 else [names[i - 1]]}
            for i, (name, status) in enumerate(zip(names, statuses))]


class SuccessorTests(unittest.TestCase):
    def check(self, tasks: list[dict], ids: object, owner: str = "measure") -> list[str]:
        return release.successor_errors(tasks, owner, ids, "scale")

    def test_new_suffix_has_no_hardcoded_measurement_id(self) -> None:
        tasks = chain("done", "todo", "todo", "todo", "todo")
        self.assertEqual(self.check(tasks, ["repair", "retest", "review"]), [])

    def test_completed_repair_prefix_does_not_invalidate_measured_miss(self) -> None:
        tasks = chain("done", "done", "in_progress", "todo", "todo")
        self.assertEqual(self.check(tasks, ["repair", "retest", "review"]), [])

    def test_single_final_review_is_valid_after_retest(self) -> None:
        tasks = chain("done", "done", "done", "todo", "todo")
        self.assertEqual(self.check(tasks, ["review"], "retest"), [])

    def test_newly_inserted_repair_cannot_be_omitted(self) -> None:
        tasks = chain("done", "todo", "todo", "todo", "todo")
        self.assertTrue(self.check(tasks, ["review"]))

    def test_duplicate_and_unknown_successors_are_rejected(self) -> None:
        tasks = chain("done", "todo", "todo", "todo", "todo")
        for ids in (["repair", "repair", "review"], ["unknown"], None):
            with self.subTest(ids=ids):
                self.assertTrue(self.check(tasks, ids))

    def test_broken_dependency_and_scale_edge_are_rejected(self) -> None:
        for index in (2, 4):
            tasks = chain("done", "todo", "todo", "todo", "todo")
            tasks[index]["depends_on"] = ["measure"]
            self.assertTrue(self.check(tasks, ["repair", "retest", "review"]))

    def test_done_after_unfinished_and_deferred_are_rejected(self) -> None:
        for statuses in (("done", "todo", "done", "todo", "todo"),
                         ("done", "deferred", "todo", "todo", "todo")):
            self.assertTrue(self.check(chain(*statuses), ["repair", "retest", "review"]))

    def test_goal_miss_without_remaining_repair_or_review_is_rejected(self) -> None:
        tasks = [{"id": "measure", "status": "done", "depends_on": []},
                 {"id": "scale", "status": "todo", "depends_on": ["measure"]}]
        self.assertTrue(self.check(tasks, []))

    def test_completed_chain_cannot_release_a_goal_miss(self) -> None:
        tasks = chain("done", "done", "done", "done", "todo")
        self.assertTrue(self.check(tasks, ["repair", "retest", "review"]))


if __name__ == "__main__":
    unittest.main()
