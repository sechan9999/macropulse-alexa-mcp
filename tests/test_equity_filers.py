"""
tests/test_equity_filers.py
─────────────────────────────────────────────────────────────────
Two filers the equity report failed on in the first forward-log record (2026-10-16):
  * JNJ: a 52/53-week fiscal year ending on the Sunday nearest Dec 31, so FY2022 ended 2023-01-01 and
    FY2023 on 2023-12-31. Both were labelled "2023" and split_adjust_eps crashed on the duplicate label.
  * XOM: the ticker moved to a new holding company (CIK 0002115436) that has only 10-Qs so far; the
    10-K history is under Exxon Mobil Corp (CIK 0000034088).
Offline: SEC facts and prices are injected.
"""
import copy
import unittest
from unittest import mock

import pandas as pd

from src.equity_report.engine import data_us, synthetic


def _dur(start, end, val, filed, form="10-K"):
    return {"start": start, "end": end, "val": val, "filed": filed, "form": form, "fp": "FY"}


def _five_two_week_facts():
    """Annual rows for a company whose fiscal year ends on the Sunday nearest Dec 31 (like JNJ)."""
    years = [("2021-01-04", "2022-01-02", 78.7, 7.81, "2022-02-17"),
             ("2022-01-03", "2023-01-01", 80.0, 6.73, "2023-02-16"),
             ("2023-01-02", "2023-12-31", 85.2, 13.72, "2024-02-16"),
             ("2024-01-01", "2024-12-29", 88.8, 5.79, "2025-02-13"),
             ("2024-12-30", "2025-12-28", 94.2, 11.03, "2026-02-11")]
    return {"facts": {
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            {"end": "2026-02-01", "val": 2400.0, "filed": "2026-02-11", "form": "10-K"}]}}},
        "us-gaap": {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
                _dur(s, e, rev, f) for s, e, rev, _, f in years]}},
            "EarningsPerShareDiluted": {"units": {"USD/shares": [
                _dur(s, e, eps, f) for s, e, _, eps, f in years]}},
        }}}


class TestFiftyTwoWeekYears(unittest.TestCase):
    def test_years_ending_in_early_january_belong_to_the_year_before(self):
        self.assertEqual(data_us.fiscal_year_label("2023-01-01"), "2022")
        self.assertEqual(data_us.fiscal_year_label("2022-01-02"), "2021")
        self.assertEqual(data_us.fiscal_year_label("2023-12-31"), "2023")
        self.assertEqual(data_us.fiscal_year_label("2025-09-27"), "2025")     # September year ends unchanged
        self.assertEqual(data_us.fiscal_year_label("2026-01-31"), "2026")     # a January year end (WMT) unchanged

    def test_annual_table_has_one_row_per_fiscal_year(self):
        fin = data_us.annual_table(_five_two_week_facts())
        self.assertEqual(list(fin.index), ["2021", "2022", "2023", "2024", "2025"])
        self.assertEqual(fin.loc["2022", "fy_end"], "2023-01-01")
        self.assertEqual(fin.loc["2023", "fy_end"], "2023-12-31")

    def test_split_adjustment_runs_on_52_53_week_years(self):
        facts = _five_two_week_facts()
        splits = pd.Series([2.0], index=[pd.Timestamp("2001-06-13")])        # an old split, as JNJ has
        fin = data_us.split_adjust_eps(data_us.annual_table(facts), facts, splits)
        self.assertEqual(fin.loc["2023", "eps_diluted"], 13.72)              # no split after the filing

    def test_duplicate_fiscal_years_are_reported_not_merged(self):
        facts = _five_two_week_facts()
        facts["facts"]["us-gaap"]["RevenueFromContractWithCustomerExcludingAssessedTax"]["units"]["USD"].append(
            _dur("2025-12-29", "2026-12-20", 1.0, "2027-02-10"))             # a second year ending in 2026
        facts["facts"]["us-gaap"]["RevenueFromContractWithCustomerExcludingAssessedTax"]["units"]["USD"].append(
            _dur("2025-11-30", "2026-11-29", 1.0, "2027-02-11"))
        with self.assertRaises(data_us.DataUnavailable):
            data_us.annual_table(facts)


SUCCESSOR, PREDECESSOR = "0009999999", "0001234567"


def _successor_facts():
    """The fixture company after a reorganization: the new registrant has filed only the 10-Q."""
    facts = copy.deepcopy(synthetic.company_facts_fixture())
    for concepts in facts["facts"].values():
        for body in concepts.values():
            for unit, rows in body["units"].items():
                body["units"][unit] = [r for r in rows if r.get("form") == "10-Q"]
    facts["facts"]["dei"] = copy.deepcopy(synthetic.company_facts_fixture()["facts"]["dei"])
    return facts


def _http(successor, predecessor):
    def get(url):
        if url.endswith("company_tickers.json"):
            return {"0": {"cik_str": int(SUCCESSOR), "ticker": "NEWCO", "title": "NEWCO HOLDINGS"}}
        if SUCCESSOR in url:
            return successor
        if PREDECESSOR in url:
            return predecessor
        raise AssertionError(url)
    return get


class TestSuccessorRegistrant(unittest.TestCase):
    def setUp(self):
        data_us._mem.clear()

    def test_registrant_without_a_10k_gets_a_plain_reason(self):
        with self.assertRaisesRegex(data_us.UnsupportedFiler, "not filed a 10-K yet"):
            data_us.annual_table(_successor_facts())

    def test_predecessor_history_is_used_for_a_mapped_successor(self):
        full = synthetic.company_facts_fixture()
        with mock.patch.dict(data_us.PREDECESSOR_CIK, {SUCCESSOR: PREDECESSOR}):
            b = data_us.load_bundle("NEWCO", price_fetch=lambda t: synthetic.price_frame(seed=3),
                                    http_get=_http(_successor_facts(), full), with_positioning=False,
                                    splits_fetch=lambda t: None)
        pd.testing.assert_frame_equal(b["financials"], data_us.annual_table(full))
        self.assertEqual(b["ttm"]["revenue"], data_us.ttm(full, data_us.annual_table(full))["revenue"])
        self.assertEqual(b["shares"], 1000.0)
        self.assertIn(f"predecessor CIK {PREDECESSOR}", b["sources"][1])

    def test_unmapped_successor_still_fails(self):
        with self.assertRaises(data_us.UnsupportedFiler):
            data_us.load_bundle("NEWCO", price_fetch=lambda t: synthetic.price_frame(seed=3),
                                http_get=_http(_successor_facts(), synthetic.company_facts_fixture()),
                                with_positioning=False, splits_fetch=lambda t: None)

    def test_merge_leaves_both_inputs_unchanged(self):
        succ, pred = _successor_facts(), synthetic.company_facts_fixture()
        before = (copy.deepcopy(succ), copy.deepcopy(pred))
        data_us.merge_facts(succ, pred)
        self.assertEqual((succ, pred), before)

    def test_xom_mapping(self):
        self.assertEqual(data_us.PREDECESSOR_CIK["0002115436"], "0000034088")


if __name__ == "__main__":
    unittest.main()
