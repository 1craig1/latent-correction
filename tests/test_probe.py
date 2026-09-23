import unittest

from same_input_probe import Item, parse_choice, question_prompt, summarize
from pathlib import Path


class ProbeTests(unittest.TestCase):
    def test_parse_choice_rejects_ambiguous_rationale(self):
        self.assertEqual(parse_choice("C", 4), "C")
        self.assertEqual(parse_choice("The answer is B.", 4), "B")
        self.assertIsNone(parse_choice("A person moves and C might be right", 4))

    def test_summary_separates_agreement_and_correctness(self):
        items = [Item("q1", Path("clip.mp4"), "?", ("a", "b"), "A"),
                 Item("q2", Path("clip.mp4"), "?", ("a", "b"), "A")]
        rows = []
        for item_id, answers in (("q1", ("A", "B")), ("q2", ("B", "B"))):
            for repeat, answer in enumerate(answers):
                rows.append({"item_id": item_id, "mode": "sample", "repeat": repeat,
                             "parsed_answer": answer, "raw_answer": answer})
        summary = summarize(items, rows, 0, 2)["modes"]["sample"]
        self.assertEqual(summary["disagreement_fraction"], 0.5)
        self.assertEqual(summary["mean_correct_fraction"], 0.25)
        self.assertTrue(summary["items"][1]["consistently_wrong"])


if __name__ == "__main__":
    unittest.main()
