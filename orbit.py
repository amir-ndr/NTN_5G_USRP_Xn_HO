#!/usr/bin/env python3
"""
orbit.py — Skyfield-based orbital geometry engine (dynamic constellation mode)

Loads ALL Starlink satellites from a TLE file. src and tgt satellites are
mutable — the controller promotes tgt→src and scans for the next tgt at each
handover, so every HO targets a satellite that is genuinely above the UE.

Provides:
  - ISL propagation delay  (srcSat ↔ tgtSat)
  - Sat-to-ground delay    (srcSat ↔ TN backhaul)
  - Elevation of both satellites as seen from the UE  (drives HO trigger)
  - Elevation of both satellites as seen from the TN ground station
  - best_visible_sat()  — scan full constellation for the best handover target
  - two_best_visible()  — return top-2 visible satellites for initial pairing

Speed of light used: 299,792.458 km/s (vacuum)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from skyfield.api import load, wgs84, EarthSatellite

C_KM_S = 299_792.458


@dataclass
class OrbitalState:
    """Snapshot of geometry at a single moment."""
    timestamp:            float   # Unix time (may be simulated)
    src_name:             str     # current source satellite name
    tgt_name:             str     # current target satellite name
    isl_km:               float   # distance srcSat ↔ tgtSat (km)
    isl_delay_ms:         float   # one-way ISL propagation delay (ms)
    sat_gnd_km:           float   # distance srcSat ↔ TN ground station (km)
    sat_gnd_delay_ms:     float   # one-way sat-to-ground delay (ms)
    src_elevation_deg:    float   # srcSat elevation from TN ground station (deg)
    tgt_elevation_deg:    float   # tgtSat elevation from TN ground station (deg)
    ue_src_elevation_deg: float   # srcSat elevation from UE — drives HO trigger
    ue_tgt_elevation_deg: float   # tgtSat elevation from UE

    def __str__(self) -> str:
        return (
            f"src={self.src_name}  tgt={self.tgt_name}  "
            f"ISL={self.isl_km:.1f} km ({self.isl_delay_ms:.2f} ms)  "
            f"SatGnd={self.sat_gnd_km:.1f} km ({self.sat_gnd_delay_ms:.2f} ms)  "
            f"UE→src={self.ue_src_elevation_deg:.1f}°  "
            f"UE→tgt={self.ue_tgt_elevation_deg:.1f}°"
        )


class OrbitalEngine:
    """
    Computes time-varying geometry between two Starlink satellites,
    a fixed TN ground station, and a UE position.

    src and tgt are updated dynamically at handover time via update_pair().

    Parameters
    ----------
    tle_path        : path to a TLE file (all satellites are loaded)
    ground_lat_deg  : TN ground station latitude (°N)
    ground_lon_deg  : TN ground station longitude (°E)
    ground_alt_m    : TN ground station altitude (m)
    ue_lat_deg      : UE latitude (°N)
    ue_lon_deg      : UE longitude (°E)
    ue_alt_m        : UE altitude (m)
    src_sat_name    : optional fixed source satellite name; empty → auto-select later
    tgt_sat_name    : optional fixed target satellite name; empty → auto-select later
    """

    def __init__(
        self,
        tle_path:       str,
        ground_lat_deg: float,
        ground_lon_deg: float,
        ground_alt_m:   float = 0.0,
        ue_lat_deg:     float | None = None,
        ue_lon_deg:     float | None = None,
        ue_alt_m:       float = 0.0,
        src_sat_name:   str = "",
        tgt_sat_name:   str = "",
    ):
        self._ts = load.timescale()

        self._ground = wgs84.latlon(
            latitude_degrees  = ground_lat_deg,
            longitude_degrees = ground_lon_deg,
            elevation_m       = ground_alt_m,
        )

        ue_lat = ue_lat_deg if ue_lat_deg is not None else ground_lat_deg
        ue_lon = ue_lon_deg if ue_lon_deg is not None else ground_lon_deg
        self._ue = wgs84.latlon(
            latitude_degrees  = ue_lat,
            longitude_degrees = ue_lon,
            elevation_m       = ue_alt_m,
        )

        print(f"[Orbit] Loading TLE file …", end=" ", flush=True)
        self._all_sats: list[EarthSatellite] = load.tle_file(tle_path, ts=self._ts)
        print(f"{len(self._all_sats)} satellites loaded")

        self._src: EarthSatellite | None = None
        self._tgt: EarthSatellite | None = None

        if src_sat_name.strip():
            self._src = self._find(self._all_sats, src_sat_name, tle_path)
        if tgt_sat_name.strip():
            self._tgt = self._find(self._all_sats, tgt_sat_name, tle_path)

        print(f"[Orbit] TN      : {ground_lat_deg:.4f}°N  {ground_lon_deg:.4f}°E  {ground_alt_m:.0f} m")
        ue_note = "  (same as TN)" if (ue_lat == ground_lat_deg and ue_lon == ground_lon_deg) else ""
        print(f"[Orbit] UE      : {ue_lat:.4f}°N  {ue_lon:.4f}°E  {ue_alt_m:.0f} m{ue_note}")
        if self._src:
            print(f"[Orbit] srcSat  : {self._src.name.strip()}  (fixed via config)")
        if self._tgt:
            print(f"[Orbit] tgtSat  : {self._tgt.name.strip()}  (fixed via config)")
        if not self._src or not self._tgt:
            print(f"[Orbit] Pair mode: dynamic — will auto-select from full constellation")

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def src_sat(self) -> EarthSatellite | None:
        return self._src

    @property
    def tgt_sat(self) -> EarthSatellite | None:
        return self._tgt

    # ── Dynamic pair management ────────────────────────────────────────────────

    def update_pair(self, src: EarthSatellite, tgt: EarthSatellite) -> None:
        """Replace the active src/tgt satellite pair (called at each HO)."""
        self._src = src
        self._tgt = tgt

    def best_visible_sat(
        self,
        unix_time:     float,
        exclude_names: set[str] | None = None,
        min_el:        float = 25.0,
    ) -> tuple[EarthSatellite, float] | None:
        """
        Scan ALL loaded satellites and return (satellite, elevation_deg) for the
        one with the highest UE elevation, subject to:
          - elevation > min_el
          - name not in exclude_names (case-insensitive)

        Returns None if no candidate qualifies.

        Runtime: ~1–5 s for a full Starlink constellation (one-time per HO).
        """
        t    = self._ts.from_datetime(datetime.fromtimestamp(unix_time, tz=timezone.utc))
        excl = {n.strip().upper() for n in (exclude_names or [])}
        best_sat: EarthSatellite | None = None
        best_el  = min_el  # threshold doubles as initial bound

        for sat in self._all_sats:
            if sat.name.strip().upper() in excl:
                continue
            try:
                el = (sat - self._ue).at(t).altaz()[0].degrees
                if el > best_el:
                    best_el  = el
                    best_sat = sat
            except Exception:
                continue

        return (best_sat, best_el) if best_sat is not None else None

    def two_best_visible(
        self,
        unix_time: float,
        min_el:    float = 25.0,
    ) -> tuple[EarthSatellite | None, EarthSatellite | None]:
        """
        Return the two satellites with the highest UE elevation above min_el.
        Used at startup to automatically choose the initial src/tgt pair.
        """
        t          = self._ts.from_datetime(datetime.fromtimestamp(unix_time, tz=timezone.utc))
        candidates: list[tuple[float, EarthSatellite]] = []

        for sat in self._all_sats:
            try:
                el = (sat - self._ue).at(t).altaz()[0].degrees
                if el > min_el:
                    candidates.append((el, sat))
            except Exception:
                continue

        candidates.sort(key=lambda x: x[0], reverse=True)
        first  = candidates[0][1] if len(candidates) > 0 else None
        second = candidates[1][1] if len(candidates) > 1 else None
        return first, second

    # ── Geometry snapshot ──────────────────────────────────────────────────────

    def state(self, unix_time: float | None = None) -> OrbitalState:
        """
        Compute geometry for the given Unix timestamp (or real time now).
        Raises RuntimeError if src/tgt have not been set yet.
        """
        if self._src is None or self._tgt is None:
            raise RuntimeError(
                "src/tgt satellites not set — call update_pair() or set sat names in config."
            )

        t_unix = unix_time if unix_time is not None else time.time()
        t = self._ts.from_datetime(datetime.fromtimestamp(t_unix, tz=timezone.utc))

        src_pos = self._src.at(t)
        tgt_pos = self._tgt.at(t)
        isl_km  = (src_pos - tgt_pos).distance().km

        src_topo   = (self._src - self._ground).at(t)
        sat_gnd_km = src_topo.distance().km
        src_el_gnd = src_topo.altaz()[0].degrees

        tgt_el_gnd = (self._tgt - self._ground).at(t).altaz()[0].degrees
        src_el_ue  = (self._src - self._ue).at(t).altaz()[0].degrees
        tgt_el_ue  = (self._tgt - self._ue).at(t).altaz()[0].degrees

        return OrbitalState(
            timestamp            = t_unix,
            src_name             = self._src.name.strip(),
            tgt_name             = self._tgt.name.strip(),
            isl_km               = isl_km,
            isl_delay_ms         = isl_km / C_KM_S * 1000.0,
            sat_gnd_km           = sat_gnd_km,
            sat_gnd_delay_ms     = sat_gnd_km / C_KM_S * 1000.0,
            src_elevation_deg    = src_el_gnd,
            tgt_elevation_deg    = tgt_el_gnd,
            ue_src_elevation_deg = src_el_ue,
            ue_tgt_elevation_deg = tgt_el_ue,
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _find(sats: list[EarthSatellite], name: str, path: str) -> EarthSatellite:
        name_up = name.strip().upper()
        for s in sats:
            if s.name.strip().upper() == name_up:
                return s
        candidates = [s for s in sats if name_up in s.name.strip().upper()]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            names = ", ".join(s.name for s in candidates[:5])
            raise ValueError(
                f"Ambiguous satellite name '{name}' — matches: {names}\n"
                f"Be more specific in controller.py"
            )
        raise ValueError(f"Satellite '{name}' not found in {path}")
