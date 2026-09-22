"""
tests/test_aws_datalake.py — S3 data-lake service (src/aws_datalake.py).

No AWS calls: the S3 client is replaced by an in-memory fake so the Parquet
round-trip (save -> load, date index, sort order, fail-soft) is exercised offline.
"""
import io
import unittest
from unittest import mock

import pandas as pd

import src.aws_datalake as dl


class _FakeS3:
    """Minimal in-memory stand-in for a boto3 S3 client."""
    def __init__(self):
        self.store = {}

    def put_object(self, Bucket, Key, Body):
        self.store[(Bucket, Key)] = Body

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.store:
            from botocore.exceptions import ClientError
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.store[(Bucket, Key)])}

    def list_objects_v2(self, Bucket, Prefix=""):
        keys = [k for (b, k) in self.store if b == Bucket and k.startswith(Prefix)]
        return {"Contents": [{"Key": k} for k in keys]}


class TestAwsDatalake(unittest.TestCase):
    def setUp(self):
        self.fake = _FakeS3()
        patcher = mock.patch.object(dl, "get_client", lambda bucket=None: (self.fake, "test-bucket", True))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_unconfigured_is_standby_and_fails_soft(self):
        # When no bucket/creds are configured, get_client reports not-ok and every
        # operation degrades quietly (Standby) without raising.
        with mock.patch.object(dl, "get_client", lambda bucket=None: (None, "", False)):
            self.assertFalse(dl.get_datalake_status()["connected"])
            self.assertIsNone(dl.load_macro_from_datalake())
            self.assertFalse(dl.save_macro_to_datalake(pd.DataFrame({"a": [1]})))
        with mock.patch.dict("os.environ", {"MACROPULSE_S3_BUCKET": ""}):
            self.assertFalse(dl.is_datalake_available())

    def test_macro_roundtrip_preserves_dates_and_values(self):
        idx = pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-02"])
        df = pd.DataFrame({"sp500": [4000.0, 4100.0, 4050.0],
                           "credit_spread": [1.1, 1.3, 1.2]}, index=idx)
        df.index.name = "date"
        self.assertTrue(dl.save_macro_to_datalake(df))
        got = dl.load_macro_from_datalake()
        self.assertIsNotNone(got)
        self.assertEqual(list(got.index.strftime("%Y-%m-%d")),
                         ["2026-01-01", "2026-01-02", "2026-01-03"])  # sorted by date
        self.assertNotIn("updated_at", got.columns)
        self.assertAlmostEqual(got.loc["2026-01-03", "sp500"], 4100.0)

    def test_quant_roundtrip_sorts_by_score(self):
        df = pd.DataFrame({"ticker": ["AAA", "BBB"], "score": [10.0, 90.0],
                           "signal": ["HOLD", "BUY"]})
        self.assertTrue(dl.save_quant_signals_to_datalake(df))
        got = dl.load_quant_signals_from_datalake()
        self.assertIsNotNone(got)
        self.assertEqual(list(got["ticker"]), ["BBB", "AAA"])  # highest score first
        self.assertIn("scan_date", got.columns)

    def test_status_lists_marts(self):
        dl.save_macro_to_datalake(pd.DataFrame({"sp500": [1.0]},
                                  index=pd.to_datetime(["2026-01-01"])))
        status = dl.get_datalake_status()
        self.assertTrue(status["connected"])
        self.assertTrue(any(t.endswith(".parquet") for t in status["tables"]))


if __name__ == "__main__":
    unittest.main()
