"""QuantGPT Cloud API client.

Uploads locally-computed factor values to quant-gpt.com for independent
validation, track-record tracking, and public attestation.

Authentication: email/password login, auto-refreshes JWT on 401.
"""

import logging
import os
import time

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

_ENV_EMAIL = "QUANTGPT_CLOUD_EMAIL"
_ENV_PASSWORD = "QUANTGPT_CLOUD_PASSWORD"
_ENV_BASE_URL = "QUANTGPT_CLOUD_URL"
_DEFAULT_BASE_URL = "https://quant-gpt.com"
_UPLOAD_BATCH_SIZE = 30
_VALID_UNIVERSES = {"hs300", "csi500", "csi1000"}


def _convert_symbol(sym: str) -> str:
    """sh.600519 -> 600519.SH, sz.000001 -> 000001.SZ"""
    if "." in sym and sym[:2] in ("sh", "sz"):
        prefix, code = sym.split(".", 1)
        return f"{code}.{prefix.upper()}"
    return sym


def _convert_day(day: dict) -> dict:
    return {
        "date": str(day["date"]).split(" ")[0],
        "values": {_convert_symbol(k): v for k, v in day["values"].items()},
    }


def is_configured() -> bool:
    return bool(os.environ.get(_ENV_EMAIL) and os.environ.get(_ENV_PASSWORD))


def get_cloud_url() -> str:
    return os.environ.get(_ENV_BASE_URL, _DEFAULT_BASE_URL).rstrip("/")


class CloudClient:
    def __init__(
        self,
        email: str | None = None,
        password: str | None = None,
        base_url: str | None = None,
    ):
        self.email = email or os.environ.get(_ENV_EMAIL, "")
        self.password = password or os.environ.get(_ENV_PASSWORD, "")
        self.base_url = (base_url or get_cloud_url()).rstrip("/")
        self._session: requests.Session | None = None
        self._token: str = ""
        self._token_expires: float = 0

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.trust_env = False
            retry = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
            adapter = HTTPAdapter(max_retries=retry)
            self._session.mount("https://", adapter)
            self._session.mount("http://", adapter)
            self._session.headers.update({
                "Content-Type": "application/json",
                "User-Agent": "QuantGPT/1.0",
            })
        return self._session

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _ensure_auth(self) -> None:
        if self._token and time.time() < self._token_expires:
            return
        self._authenticate()

    def _authenticate(self) -> None:
        s = self._get_session()
        resp = s.post(
            self._url("/api/v1/auth/login"),
            json={"email": self.email, "password": self.password},
        )
        if not resp.ok:
            raise CloudAPIError(f"Cloud login failed (HTTP {resp.status_code}): {resp.text[:200]}")
        data = resp.json()
        self._token = data["access_token"]
        self._token_expires = time.time() + 23 * 3600
        s.headers["Authorization"] = f"Bearer {self._token}"
        logger.info("Authenticated with QuantGPT Cloud as %s", self.email)

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        self._ensure_auth()
        s = self._get_session()
        resp = getattr(s, method)(self._url(path), **kwargs)
        if resp.status_code == 401:
            self._token = ""
            self._ensure_auth()
            resp = getattr(s, method)(self._url(path), **kwargs)
        return resp

    def _raise_for_error(self, resp: requests.Response, action: str) -> None:
        if resp.ok:
            return
        try:
            detail = resp.json().get("detail", resp.text[:200])
        except Exception:
            detail = resp.text[:200]
        raise CloudAPIError(f"{action} failed (HTTP {resp.status_code}): {detail}")

    def create_factor(
        self,
        name: str,
        universe: str,
        expression: str | None = None,
        description: str | None = None,
        claimed_ic_mean: float | None = None,
        claimed_ic_ir: float | None = None,
    ) -> dict:
        payload: dict = {"name": name, "universe": universe}
        if expression:
            payload["expression"] = expression
        if description:
            payload["description"] = description
        if claimed_ic_mean is not None:
            payload["claimed_ic_mean"] = claimed_ic_mean
        if claimed_ic_ir is not None:
            payload["claimed_ic_ir"] = claimed_ic_ir

        resp = self._request("post", "/api/v1/factors", json=payload)
        self._raise_for_error(resp, "create_factor")
        return resp.json()

    def upload_values(self, factor_id: str, data: list[dict]) -> dict:
        resp = self._request("post", f"/api/v1/factors/{factor_id}/values", json={"data": data})
        self._raise_for_error(resp, "upload_values")
        return resp.json()

    def upload_values_batched(self, factor_id: str, data: list[dict]) -> dict:
        total_uploaded = 0
        for i in range(0, len(data), _UPLOAD_BATCH_SIZE):
            batch = data[i : i + _UPLOAD_BATCH_SIZE]
            result = self.upload_values(factor_id, batch)
            total_uploaded += result.get("uploaded", len(batch))
        return {"uploaded": total_uploaded, "batches": (len(data) + _UPLOAD_BATCH_SIZE - 1) // _UPLOAD_BATCH_SIZE}

    def submit_factor(self, factor_id: str) -> dict:
        resp = self._request("post", f"/api/v1/factors/{factor_id}/submit")
        self._raise_for_error(resp, "submit_factor")
        return resp.json()

    def get_factor(self, factor_id: str) -> dict:
        resp = self._request("get", f"/api/v1/factors/{factor_id}")
        self._raise_for_error(resp, "get_factor")
        return resp.json()

    def upload_and_validate(
        self,
        name: str,
        universe: str,
        factor_values_data: list[dict],
        expression: str | None = None,
        claimed_ic_mean: float | None = None,
        claimed_ic_ir: float | None = None,
    ) -> dict:
        factor = self.create_factor(
            name=name,
            universe=universe,
            expression=expression,
            claimed_ic_mean=claimed_ic_mean,
            claimed_ic_ir=claimed_ic_ir,
        )
        factor_id = factor["id"]
        logger.info("Created cloud factor %s (%s)", factor_id, name)

        converted_data = [_convert_day(d) for d in factor_values_data]
        upload_result = self.upload_values_batched(factor_id, converted_data)
        logger.info(
            "Uploaded %d days in %d batches",
            upload_result["uploaded"],
            upload_result["batches"],
        )

        result = self.submit_factor(factor_id)
        logger.info("Cloud validation: status=%s", result.get("status"))
        return result


class CloudAPIError(Exception):
    pass


def factor_df_to_cloud_format(factor_df: pd.DataFrame) -> list[dict]:
    """Convert backtest _factor_df to Cloud upload format.

    Input:  DataFrame with columns [trade_date, stock_code, factor_value, ...]
    Output: [{date: "2024-01-02", values: {"sh.600519": 0.123}}, ...]
    """
    data = []
    for trade_date, group in factor_df.groupby("trade_date"):
        values = {}
        for _, row in group.iterrows():
            val = row["factor_value"]
            if np.isfinite(val):
                values[row["stock_code"]] = round(float(val), 6)
        if values:
            data.append({"date": str(trade_date).split(" ")[0], "values": values})
    return data


def auto_upload_to_cloud(
    expression: str,
    universe: str,
    factor_df: pd.DataFrame,
    claimed_ic_mean: float | None = None,
    claimed_ic_ir: float | None = None,
) -> dict | None:
    """Auto-upload A-grade factor to Cloud for independent validation.

    Returns Cloud validation result dict, or None if skipped/failed.
    """
    if not is_configured():
        return None
    if universe not in _VALID_UNIVERSES:
        return None

    try:
        data = factor_df_to_cloud_format(factor_df)
        if len(data) < 30:
            logger.info("Cloud auto-upload skipped: only %d trading days", len(data))
            return None

        client = CloudClient()
        result = client.upload_and_validate(
            name=expression[:80],
            universe=universe,
            factor_values_data=data,
            expression=expression,
            claimed_ic_mean=claimed_ic_mean,
            claimed_ic_ir=claimed_ic_ir,
        )
        logger.info("Cloud auto-upload completed: status=%s", result.get("status"))
        return result
    except Exception:
        logger.exception("Cloud auto-upload failed")
        return None
