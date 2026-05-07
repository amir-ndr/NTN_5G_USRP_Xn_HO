#!/usr/bin/env python3
"""
neone_ctrl.py — NE-ONE REST API client

Pushes propagation delays to two named NE-ONE links:
    Link A : ISL          (USRP1 srcSat ↔ USRP2 targetSat)
    Link B : Sat-to-ground (USRP1 srcSat ↔ USRP3 TN node)

NE-ONE REST API reference
─────────────────────────
Base URL  : http://<NE-ONE-IP>/api/v1
Auth      : HTTP Basic or Bearer token (set in __init__)
Delay PUT : PUT /links/{link_id}/impairments/delay
            Body: {"one_way_delay_ms": <float>}

If your NE-ONE version uses a different endpoint path, update
_URL_DELAY below to match.  Everything else stays the same.
"""

from __future__ import annotations

import requests
from requests.auth import HTTPBasicAuth


# ── Endpoint template — adjust to your NE-ONE firmware version ───────────────
# Most NE-ONE versions use one of these two forms:
#   /api/v1/links/{id}/impairments/delay          (newer)
#   /api/v1/network-map/links/{id}/delay          (older)
_URL_DELAY = "/api/v1/links/{link_id}/impairments/delay"


class NeoNEController:
    """
    Thin REST client for the NE-ONE network emulator.

    Parameters
    ----------
    host         : IP address or hostname of the NE-ONE unit (e.g. "192.168.1.100")
    isl_link_id  : NE-ONE link ID for the ISL pipe (string or int, from NE-ONE GUI)
    gnd_link_id  : NE-ONE link ID for the sat-to-ground pipe
    username     : NE-ONE login username  (leave "" if no auth required)
    password     : NE-ONE login password  (leave "" if no auth required)
    timeout_s    : HTTP request timeout in seconds
    dry_run      : if True, print what would be sent but don't actually send
    """

    def __init__(
        self,
        host:        str,
        isl_link_id: str | int,
        gnd_link_id: str | int,
        username:    str   = "",
        password:    str   = "",
        timeout_s:   float = 3.0,
        dry_run:     bool  = False,
    ):
        self._base    = f"http://{host}"
        self._isl_id  = str(isl_link_id)
        self._gnd_id  = str(gnd_link_id)
        self._auth    = HTTPBasicAuth(username, password) if username else None
        self._timeout = timeout_s
        self._dry_run = dry_run
        self._session = requests.Session()

        tag = " [DRY RUN]" if dry_run else ""
        print(f"[NE-ONE] host={host}  ISL link={isl_link_id}  GND link={gnd_link_id}{tag}")

    # ── Public API ────────────────────────────────────────────────────────────

    def set_isl_delay(self, delay_ms: float) -> bool:
        """Push a new one-way delay to the ISL link. Returns True on success."""
        return self._set_delay(self._isl_id, delay_ms, label="ISL")

    def set_sat_gnd_delay(self, delay_ms: float) -> bool:
        """Push a new one-way delay to the sat-to-ground link. Returns True on success."""
        return self._set_delay(self._gnd_id, delay_ms, label="SatGnd")

    def set_both(self, isl_ms: float, gnd_ms: float) -> bool:
        """Convenience: update both links in one call. Returns True if both succeed."""
        ok1 = self.set_isl_delay(isl_ms)
        ok2 = self.set_sat_gnd_delay(gnd_ms)
        return ok1 and ok2

    # ── Internals ─────────────────────────────────────────────────────────────

    def _set_delay(self, link_id: str, delay_ms: float, label: str) -> bool:
        url  = self._base + _URL_DELAY.format(link_id=link_id)
        body = {"one_way_delay_ms": round(delay_ms, 3)}

        if self._dry_run:
            print(f"[NE-ONE][DRY] PUT {url}  body={body}")
            return True

        try:
            resp = self._session.put(
                url, json=body, auth=self._auth, timeout=self._timeout
            )
            resp.raise_for_status()
            return True
        except requests.exceptions.RequestException as exc:
            print(f"[NE-ONE][WARN] {label} delay update failed: {exc}")
            return False
