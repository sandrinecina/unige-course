"""Client for the Zefix public REST API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import requests
from requests import Response, Session
from requests.auth import HTTPBasicAuth

from ._utils import require_setting
from .config import get_settings


DEFAULT_BASE_URL = "https://www.zefix.admin.ch/ZefixPublicREST"
DEFAULT_TIMEOUT = 30.0


class ZefixError(RuntimeError):
    """Error raised when a Zefix API call fails."""

    def __init__(self, message: str, *, status_code: int, detail: Any | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


@dataclass(slots=True)
class ZefixClient:
    """Thin wrapper around the Zefix REST API with basic auth."""

    username: str
    password: str
    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT
    session: Session | None = field(default=None, repr=False)
    _session: Session = field(init=False, repr=False)
    _auth: HTTPBasicAuth = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        object.__setattr__(self, "_session", self.session or requests.Session())
        if not hasattr(self._session, "headers") or self._session.headers is None:
            self._session.headers = {}  # type: ignore[assignment]
        self._session.headers.setdefault("Accept", "application/json")
        auth = HTTPBasicAuth(self.username, self.password)
        object.__setattr__(self, "_auth", auth)
        self._session.auth = auth

    @property
    def auth(self) -> HTTPBasicAuth:
        return self._auth

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> ZefixClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Mapping[str, Any] | None = None,
    ) -> Any:
        if not path.startswith("/"):
            path = "/" + path
        url = f"{self.base_url}{path}"

        try:
            response = self._session.request(
                method,
                url,
                params=params,
                json=json,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:  # pragma: no cover - network failures rare in tests
            raise ZefixError("Zefix API request failed", status_code=0, detail=str(exc)) from exc

        return self._handle_response(response)

    def _handle_response(self, response: Response) -> Any:
        if response.ok:
            if response.status_code == 204 or not response.content:
                return None
            if "application/json" in response.headers.get("Content-Type", ""):
                try:
                    return response.json()
                except ValueError as exc:
                    raise ZefixError(
                        "Failed to decode JSON from Zefix response",
                        status_code=response.status_code,
                        detail=response.text,
                    ) from exc
            return response.text

        detail: Any | None = None
        message = f"Zefix API responded with status {response.status_code}"

        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            try:
                detail = response.json()
                error = detail.get("error") if isinstance(detail, dict) else None
                if isinstance(error, dict):
                    error_type = error.get("type")
                    error_message = error.get("message")
                    if error_type and error_message:
                        message = f"{error_type}: {error_message}"
                    elif error_message:
                        message = error_message
            except ValueError:
                detail = response.text
        else:
            detail = response.text

        raise ZefixError(message, status_code=response.status_code, detail=detail)

    def _ensure_list(self, data: Any, *, context: str) -> list[dict[str, Any]]:
        if data is None:
            return []
        if isinstance(data, list):
            return data
        raise ZefixError(
            f"Unexpected response payload for {context}",
            status_code=200,
            detail=data,
        )

    def _ensure_dict(self, data: Any, *, context: str) -> dict[str, Any]:
        if data is None:
            return {}
        if isinstance(data, dict):
            return data
        raise ZefixError(
            f"Unexpected response payload for {context}",
            status_code=200,
            detail=data,
        )

    def search_companies(
        self,
        *,
        name: str,
        legal_form_id: int | None = None,
        legal_form_uid: str | None = None,
        registry_of_commerce_id: int | None = None,
        legal_seat_id: int | None = None,
        canton: str | None = None,
        active_only: bool | None = None,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"name": name}
        if legal_form_id is not None:
            payload["legalFormId"] = legal_form_id
        if legal_form_uid is not None:
            payload["legalFormUid"] = legal_form_uid
        if registry_of_commerce_id is not None:
            payload["registryOfCommerceId"] = registry_of_commerce_id
        if legal_seat_id is not None:
            payload["legalSeatId"] = legal_seat_id
        if canton is not None:
            payload["canton"] = canton
        if active_only is not None:
            payload["activeOnly"] = active_only

        result = self._request("POST", "/api/v1/company/search", json=payload)
        return self._ensure_list(result, context="company search")

    def get_company_by_uid(self, uid: str) -> list[dict[str, Any]]:
        result = self._request("GET", f"/api/v1/company/uid/{uid}")
        return self._ensure_list(result, context="company uid lookup")

    def get_company_by_ehraid(self, ehraid: int) -> list[dict[str, Any]]:
        result = self._request("GET", f"/api/v1/company/ehraid/{ehraid}")
        return self._ensure_list(result, context="company ehraid lookup")

    def get_company_by_chid(self, chid: str) -> list[dict[str, Any]]:
        result = self._request("GET", f"/api/v1/company/chid/{chid}")
        return self._ensure_list(result, context="company chid lookup")

    def get_sogc_publication(self, sogc_id: int) -> dict[str, Any]:
        result = self._request("GET", f"/api/v1/sogc/{sogc_id}")
        return self._ensure_dict(result, context="sogc publication")

    def get_sogc_publications_by_date(self, date: str) -> list[dict[str, Any]]:
        result = self._request("GET", f"/api/v1/sogc/bydate/{date}")
        return self._ensure_list(result, context="sogc by date")

    def list_registries_of_commerce(self) -> list[dict[str, Any]]:
        result = self._request("GET", "/api/v1/registryOfCommerce")
        return self._ensure_list(result, context="registry of commerce list")

    def get_registry_of_commerce_by_bfs_id(self, bfs_id: str) -> dict[str, Any]:
        result = self._request("GET", f"/api/v1/registryOfCommerce/byBfsCommunityId/{bfs_id}")
        return self._ensure_dict(result, context="registry of commerce by bfs id")

    def list_legal_forms(self) -> list[dict[str, Any]]:
        result = self._request("GET", "/api/v1/legalForm")
        return self._ensure_list(result, context="legal forms")

    def list_communities(self) -> list[dict[str, Any]]:
        result = self._request("GET", "/api/v1/community")
        return self._ensure_list(result, context="communities list")


def get_zefix_client(
    *,
    session: Session | None = None,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> ZefixClient:
    """Create a `ZefixClient` using environment settings."""

    settings = get_settings()
    username = require_setting(settings.zefix_username, name="ZEFIX_USERNAME")
    password = require_setting(settings.zefix_password, name="ZEFIX_PASSWORD")

    return ZefixClient(
        username=username,
        password=password,
        base_url=base_url,
        timeout=timeout,
        session=session,
    )

