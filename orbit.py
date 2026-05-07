#!/usr/bin/env python3
"""
orbit.py — Skyfield-based orbital geometry engine

Loads two Starlink satellites from a local TLE file, a fixed ground station
(TN backhaul anchor), and a UE position (HO trigger point).

Provides:
  - ISL propagation delay (srcSat ↔ trgSat)
  - Sat-to-ground delay (srcSat ↔ TN)
  - Elevation of both satellites as seen from the UE (drives HO trigger)
  - Elevation of both satellites as seen from the TN ground station

Speed of light used: 299,792.458 km/s (vacuum)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from skyfield.api import load, wgs84, EarthSatellite

C_KM_S = 299_792.458   # speed of light in km/s


@dataclass
class OrbitalState:
    """Snapshot of geometry at a single moment."""
    timestamp:            float   # Unix time (seconds, may be simulated)
    isl_km:               float   # distance srcSat ↔ trgSat (km)
    isl_delay_ms:         float   # one-way ISL propagation delay (ms)
    sat_gnd_km:           float   # distance srcSat ↔ TN ground station (km)
    sat_gnd_delay_ms:     float   # one-way sat-to-ground delay (ms)
    src_elevation_deg:    float   # srcSat elevation from TN ground station (deg)
    tgt_elevation_deg:    float   # trgSat elevation from TN ground station (deg)
    ue_src_elevation_deg: float   # srcSat elevation from UE — drives HO trigger
    ue_tgt_elevation_deg: float   # trgSat elevation from UE

    def __str__(self) -> str:
        return (
            f"ISL={self.isl_km:.1f} km ({self.isl_delay_ms:.2f} ms)  "
            f"SatGnd={self.sat_gnd_km:.1f} km ({self.sat_gnd_delay_ms:.2f} ms)  "
            f"UE→src={self.ue_src_elevation_deg:.1f}°  "
            f"UE→tgt={self.ue_tgt_elevation_deg:.1f}°"
        )


class OrbitalEngine:
    """
    Computes time-varying geometry between two Starlink satellites,
    a fixed TN ground station, and a UE position.

    Parameters
    ----------
    tle_path        : path to a TLE file
    src_sat_name    : source satellite name (as in TLE file)
    tgt_sat_name    : target satellite name (as in TLE file)
    ground_lat_deg  : TN ground station latitude (°N)
    ground_lon_deg  : TN ground station longitude (°E)
    ground_alt_m    : TN ground station altitude (m)
    ue_lat_deg      : UE latitude (°N)  — HO trigger geometry
    ue_lon_deg      : UE longitude (°E)
    ue_alt_m        : UE altitude (m)
    """

    def __init__(
        self,
        tle_path:       str,
        src_sat_name:   str,
        tgt_sat_name:   str,
        ground_lat_deg: float,
        ground_lon_deg: float,
        ground_alt_m:   float = 0.0,
        ue_lat_deg:     float | None = None,
        ue_lon_deg:     float | None = None,
        ue_alt_m:       float = 0.0,
    ):
        self._ts = load.timescale()

        self._ground = wgs84.latlon(
            latitude_degrees  = ground_lat_deg,
            longitude_degrees = ground_lon_deg,
            elevation_m       = ground_alt_m,
        )

        # UE defaults to same position as ground station if not provided
        ue_lat = ue_lat_deg if ue_lat_deg is not None else ground_lat_deg
        ue_lon = ue_lon_deg if ue_lon_deg is not None else ground_lon_deg
        self._ue = wgs84.latlon(
            latitude_degrees  = ue_lat,
            longitude_degrees = ue_lon,
            elevation_m       = ue_alt_m,
        )

        sats = self._load_tle(tle_path)
        self._src = self._find(sats, src_sat_name, tle_path)
        self._tgt = self._find(sats, tgt_sat_name, tle_path)

        print(f"[Orbit] srcSat  : {self._src.name}")
        print(f"[Orbit] tgtSat  : {self._tgt.name}")
        print(f"[Orbit] TN      : {ground_lat_deg:.4f}°N  {ground_lon_deg:.4f}°E  {ground_alt_m:.0f} m")
        print(f"[Orbit] UE      : {ue_lat:.4f}°N  {ue_lon:.4f}°E  {ue_alt_m:.0f} m"
              + ("  (same as TN)" if ue_lat == ground_lat_deg and ue_lon == ground_lon_deg else ""))

    # ── Public API ────────────────────────────────────────────────────────────

    def state(self, unix_time: float | None = None) -> OrbitalState:
        """
        Compute and return an OrbitalState for the given Unix timestamp.
        Pass a simulated unix_time to support time-acceleration mode.
        Defaults to real time now if None.
        """
        t_unix = unix_time if unix_time is not None else time.time()
        t = self._ts.from_datetime(
            datetime.fromtimestamp(t_unix, tz=timezone.utc)
        )

        # ISL distance
        src_pos = self._src.at(t)
        tgt_pos = self._tgt.at(t)
        isl_km  = (src_pos - tgt_pos).distance().km

        # srcSat → TN ground station
        src_topo   = (self._src - self._ground).at(t)
        sat_gnd_km = src_topo.distance().km
        src_el_gnd = src_topo.altaz()[0].degrees

        # trgSat → TN ground station
        tgt_el_gnd = (self._tgt - self._ground).at(t).altaz()[0].degrees

        # srcSat → UE  (drives HO trigger)
        src_el_ue  = (self._src - self._ue).at(t).altaz()[0].degrees

        # trgSat → UE
        tgt_el_ue  = (self._tgt - self._ue).at(t).altaz()[0].degrees

        return OrbitalState(
            timestamp            = t_unix,
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
    def _load_tle(path: str) -> list[EarthSatellite]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"TLE file not found: {path}")
        ts = load.timescale()
        return load.tle_file(str(p), ts=ts)

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
