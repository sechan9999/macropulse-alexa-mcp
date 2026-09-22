"""
AWS S3 data-lake service for Macro Pulse (hf-macro-dashboard).

The AWS replacement for src/bigquery_service.py. The app writes whole computed
dataframes (macro factors, quant signals) and reads them back whole — no ad-hoc
SQL — so a columnar object in S3 is the right fit, not a query engine. Each "mart"
is a single Parquet object under s3://<bucket>/<prefix>/<name>.parquet, overwritten
in place (the equivalent of BigQuery's WRITE_TRUNCATE).

Configuration (all optional — with none set, the app runs on live yfinance/local):
    MACROPULSE_S3_BUCKET   the bucket to use; when unset the data lake is "Standby"
    MACROPULSE_S3_PREFIX   key prefix, default "macropulse"
    AWS_REGION / AWS_DEFAULT_REGION   region for the S3 client, default us-east-1
    AWS credentials come from boto3's default chain (env vars, shared config, or
    an instance/task role) — this module never handles secrets itself.
"""

import io
import os
import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd

_BOTO3_AVAILABLE = False
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    _BOTO3_AVAILABLE = True
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore
    BotoCoreError = ClientError = Exception  # type: ignore

_PREFIX = os.getenv("MACROPULSE_S3_PREFIX", "macropulse").strip("/")
_MACRO_KEY = "macro_factors.parquet"
_QUANT_KEY = "quant_signals.parquet"


def _bucket(bucket: Optional[str] = None) -> str:
    return bucket or os.getenv("MACROPULSE_S3_BUCKET", "")


def _region() -> str:
    return os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"


def _key(name: str) -> str:
    return f"{_PREFIX}/{name}" if _PREFIX else name


def is_datalake_available() -> bool:
    """True when boto3 is installed and a target bucket is configured.

    Mirrors is_bigquery_available(): a cheap gate. Real connectivity is checked in
    get_datalake_status(); reads/writes fail soft on missing credentials."""
    return _BOTO3_AVAILABLE and bool(_bucket())


def get_client(bucket: Optional[str] = None) -> Tuple[Optional[Any], str, bool]:
    """Return (s3_client, bucket, ok). ok is False if boto3 or the bucket is missing."""
    b = _bucket(bucket)
    if not _BOTO3_AVAILABLE or not b:
        return None, b, False
    try:
        return boto3.client("s3", region_name=_region()), b, True
    except Exception:
        return None, b, False


def get_datalake_status(bucket: Optional[str] = None, prefix: str = _PREFIX) -> Dict[str, Any]:
    """Check connectivity and list the mart objects under the prefix."""
    client, b, ok = get_client(bucket)
    if not ok or client is None:
        return {"connected": False, "bucket": b, "prefix": prefix, "tables": [],
                "error": "boto3 missing or MACROPULSE_S3_BUCKET not set"}
    try:
        resp = client.list_objects_v2(Bucket=b, Prefix=f"{prefix}/" if prefix else "")
        tables = [os.path.basename(o["Key"]) for o in resp.get("Contents", [])
                  if o["Key"].endswith(".parquet")]
        return {"connected": True, "bucket": b, "prefix": prefix, "tables": tables, "error": None}
    except (BotoCoreError, ClientError, Exception) as e:
        return {"connected": False, "bucket": b, "prefix": prefix, "tables": [], "error": str(e)}


def _put_parquet(df: pd.DataFrame, key: str, bucket: Optional[str] = None) -> bool:
    client, b, ok = get_client(bucket)
    if not ok or client is None:
        return False
    try:
        buf = io.BytesIO()
        df.to_parquet(buf, engine="pyarrow", index=False)
        client.put_object(Bucket=b, Key=key, Body=buf.getvalue())
        return True
    except Exception as e:
        print(f"[S3] Failed to write {key}: {e}")
        return False


def _get_parquet(key: str, bucket: Optional[str] = None) -> Optional[pd.DataFrame]:
    client, b, ok = get_client(bucket)
    if not ok or client is None:
        return None
    try:
        obj = client.get_object(Bucket=b, Key=key)
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()), engine="pyarrow")
        return df if not df.empty else None
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NoSuchBucket"):
            return None
        print(f"[S3] Failed to read {key}: {e}")
        return None
    except Exception as e:
        print(f"[S3] Failed to read {key}: {e}")
        return None


def save_macro_to_datalake(df: pd.DataFrame, bucket: Optional[str] = None) -> bool:
    """Overwrite the macro-factors mart with df (date stored as a column)."""
    upload = df.copy()
    if "date" not in upload.columns:
        upload = upload.reset_index().rename(columns={"index": "date"})
    upload["date"] = pd.to_datetime(upload["date"]).dt.date
    upload["updated_at"] = datetime.datetime.utcnow()
    for col in upload.columns:
        if upload[col].dtype == "object" and col != "date":
            upload[col] = upload[col].astype(str)
    return _put_parquet(upload, _key(_MACRO_KEY), bucket)


def load_macro_from_datalake(bucket: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load the macro-factors mart with a DatetimeIndex on `date`."""
    df = _get_parquet(_key(_MACRO_KEY), bucket)
    if df is None:
        return None
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
    if "updated_at" in df.columns:
        df = df.drop(columns=["updated_at"])
    df.index.name = "date"
    return df


def save_quant_signals_to_datalake(signals_df: pd.DataFrame, bucket: Optional[str] = None) -> bool:
    """Overwrite the quant-signals mart with the latest scan."""
    upload = signals_df.copy()
    upload["scan_date"] = datetime.date.today().isoformat()
    upload["created_at"] = datetime.datetime.utcnow()
    for col in upload.columns:
        if upload[col].dtype == "object":
            upload[col] = upload[col].astype(str)
    return _put_parquet(upload, _key(_QUANT_KEY), bucket)


def load_quant_signals_from_datalake(bucket: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load the quant-signals mart, highest score first."""
    df = _get_parquet(_key(_QUANT_KEY), bucket)
    if df is None:
        return None
    if "score" in df.columns:
        df = df.sort_values("score", ascending=False)
    return df
