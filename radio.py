#!/usr/bin/env python3
"""
radio.py — Shared radio layer for USRP X310 BPSK link

Provides:
    Packet    — structured message (magic + seq + msg_id + payload + CRC-8)
    USRP      — hardware abstraction (TX or RX role, context-manager-safe)
    Visualizer — live matplotlib plot of Barker correlation + noise histogram
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass, field

import numpy as np
import uhd

try:
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False

# ── Physical-layer constants ──────────────────────────────────────────────────

SAMPLE_RATE   = 1e6       # Hz
CENTER_FREQ   = 3.45e9    # Hz
SUBDEV        = "B:0"
ANTENNA       = "TX/RX"
SPS           = 8         # samples per symbol → symbol rate = 125 ksps
SNR_THRESHOLD = 5.0       # correlation peak / noise floor for preamble detection
COOLDOWN_S    = 0.05      # min seconds between consecutive decode attempts
MAX_PAYLOAD   = 248       # bytes of JSON per packet (8-bit frame length → 254 B frame max, minus 6 B overhead)

BARKER    = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.float32)
BARKER_UP = np.repeat(BARKER, SPS).astype(np.complex64)   # upsampled preamble reference


# ── Packet ────────────────────────────────────────────────────────────────────

_MAGIC = b"\xDE\xAD"


@dataclass
class Packet:
    """
    Structured message packet carrying a dict[str, str] payload.

    Wire format inside the BPSK frame (after Barker preamble + frame-length byte):
        MAGIC(2) | SEQ(1) | MSG_ID(1) | JSON_LEN(1) | JSON bytes | CRC8(1)

    The dict is serialized as compact JSON so it stays human-readable and
    extensible without changing any of the physical-layer code.

    Fields
    ------
    payload : dict with exactly 8 string keys and string values
    msg_id  : application-level message type (0–255)
    seq     : sequence number (0–255, wraps)
    """
    payload: dict = field(default_factory=dict)
    msg_id:  int  = 0
    seq:     int  = 0

    def encode(self) -> bytes:
        json_bytes = json.dumps(self.payload, separators=(",", ":")).encode("utf-8")
        assert len(json_bytes) <= MAX_PAYLOAD, (
            f"Payload too large ({len(json_bytes)} B > {MAX_PAYLOAD} B) — shorten your values"
        )
        # Header: SEQ(1) MSG_ID(1) JSON_LEN(1, uint8)
        body = struct.pack("BBB", self.seq & 0xFF, self.msg_id & 0xFF, len(json_bytes)) + json_bytes
        return _MAGIC + body + bytes([_crc8(body)])

    @staticmethod
    def decode(raw: bytes) -> "Packet":
        if len(raw) < len(_MAGIC) + 4:          # magic(2)+seq(1)+msg_id(1)+len(1)+crc(1)
            raise ValueError("Packet too short")
        if raw[:2] != _MAGIC:
            raise ValueError(f"Bad magic: {raw[:2].hex()}")
        seq, msg_id, json_len = struct.unpack("BBB", raw[2:5])  # 3-byte header
        end = 5 + json_len
        if len(raw) < end + 1:
            raise ValueError("Packet truncated")
        body     = raw[2:end]
        crc_recv = raw[end]
        crc_calc = _crc8(body)
        if crc_recv != crc_calc:
            raise ValueError(f"CRC mismatch (got {crc_recv:#04x}, want {crc_calc:#04x})")
        payload = json.loads(raw[5:end].decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Payload is not a JSON object")
        return Packet(payload=payload, msg_id=msg_id, seq=seq)

    def __str__(self) -> str:
        fields = "  ".join(f"{k}={v!r}" for k, v in self.payload.items())
        return f"[seq={self.seq:03d} msg_id={self.msg_id:#04x}]  {fields}"


def _crc8(data: bytes) -> int:
    """CRC-8/SMBUS polynomial 0x07."""
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) if (crc & 0x80) else (crc << 1)
            crc &= 0xFF
    return crc


# ── Modulation / demodulation internals ──────────────────────────────────────

def _build_frame(pkt: Packet) -> np.ndarray:
    """Packet → IQ samples ready for USRP TX streamer."""
    pkt_bytes = pkt.encode()
    n         = len(pkt_bytes)
    assert n <= 65535, f"Encoded packet too large: {n} B"

    # 8-bit frame length before the payload
    length_bits  = np.unpackbits(np.array([n], dtype=np.uint8)).astype(np.float32)
    payload_bits = np.unpackbits(np.frombuffer(pkt_bytes, dtype=np.uint8)).astype(np.float32)
    all_bits     = np.concatenate([length_bits, payload_bits])

    syms       = (2.0 * all_bits - 1.0).astype(np.complex64)
    frame_syms = np.concatenate([BARKER.astype(np.complex64), syms])
    samples    = np.repeat(frame_syms, SPS).astype(np.complex64)

    peak = np.max(np.abs(samples))
    return samples / peak if peak > 0 else samples


def _find_preamble(samples: np.ndarray) -> tuple[int, float, float, np.ndarray]:
    """
    Cross-correlate with BARKER_UP (magnitude → phase-agnostic).

    Returns
    -------
    peak_idx  : index of highest correlation
    peak_val  : correlation magnitude at peak
    noise_flr : mean correlation outside a guard band around the peak
    corr      : full correlation array (for visualization)
    """
    corr = np.abs(np.correlate(samples, BARKER_UP, mode="valid"))
    if corr.size == 0:
        return -1, 0.0, 0.0, corr

    peak_idx = int(np.argmax(corr))
    peak_val = float(corr[peak_idx])

    lo, hi = max(0, peak_idx - len(BARKER_UP)), min(len(corr), peak_idx + len(BARKER_UP))
    mask = np.ones(len(corr), dtype=bool)
    mask[lo:hi] = False
    noise_flr = float(np.mean(corr[mask])) if mask.any() else float(np.mean(corr))

    return peak_idx, peak_val, noise_flr, corr


def _estimate_phase(pre_samples: np.ndarray) -> float:
    """Estimate carrier phase offset from the received preamble region."""
    rx_syms  = pre_samples[SPS // 2 :: SPS][: len(BARKER)]
    ref_syms = BARKER.astype(np.complex64)
    return float(np.angle(np.dot(rx_syms, np.conj(ref_syms))))


def _demodulate(samples: np.ndarray, n_bits: int, phase: float) -> np.ndarray:
    """Phase-correct → downsample to symbol rate → hard BPSK decision."""
    corrected = samples * np.exp(-1j * phase)
    syms = corrected[SPS // 2 : SPS // 2 + n_bits * SPS : SPS][: n_bits]
    return (np.real(syms) > 0).astype(np.uint8)


# ── USRP class ────────────────────────────────────────────────────────────────

class USRP:
    """
    Hardware abstraction for a single USRP X310 in TX or RX role.

    Usage
    -----
    with USRP("192.168.20.6", role="tx", gain=10) as radio:
        while True:
            radio.transmit("Hello World")

    with USRP("192.168.10.4", role="rx", gain=20) as radio:
        pkt, info = radio.receive()
    """

    def __init__(
        self,
        addr:        str,
        role:        str,           # "tx" or "rx"
        gain:        float = 30.0,
        center_freq: float = CENTER_FREQ,
        sample_rate: float = SAMPLE_RATE,
        subdev:      str   = SUBDEV,
        antenna:     str   = ANTENNA,
        ext_ref:     bool  = True,  # True → lock to external 10 MHz (OctoClock-G)
    ):
        assert role in ("tx", "rx"), "role must be 'tx' or 'rx'"
        self.role  = role
        self._freq = center_freq
        self._rate = sample_rate
        self._seq  = 0

        dev_args   = f"addr={addr}" if addr else ""
        self._usrp = uhd.usrp.MultiUSRP(dev_args)

        if role == "tx":
            self._setup_tx(subdev, antenna, gain, ext_ref)
        else:
            self._setup_rx(subdev, antenna, gain, ext_ref)
            # Fixed 32 768-sample buffer (~33 ms @ 1 Msps).
            # Enough for one max frame (~16 K samples) with 2× headroom,
            # and small enough for numpy correlate to finish before the next overflow.
            self._recv_buf = np.zeros(32768, dtype=np.complex64)
            self._last_rx  = 0.0

    # ── Hardware setup ────────────────────────────────────────────────────────

    def _setup_tx(self, subdev, antenna, gain, ext_ref):
        u = self._usrp
        u.set_tx_subdev_spec(uhd.usrp.SubdevSpec(subdev))
        u.set_tx_rate(self._rate)
        u.set_tx_freq(uhd.types.TuneRequest(self._freq))
        u.set_tx_gain(gain)
        u.set_tx_antenna(antenna)
        self._apply_ref(ext_ref)
        print(f"[TX] rate={u.get_tx_rate()/1e6:.3f} Msps  "
              f"freq={u.get_tx_freq()/1e9:.4f} GHz  "
              f"gain={u.get_tx_gain():.1f} dB  ant={u.get_tx_antenna()}")
        self._streamer = u.get_tx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
        self._md = uhd.types.TXMetadata()
        self._md.start_of_burst = True
        self._md.end_of_burst   = False
        self._md.has_time_spec  = False

    def _setup_rx(self, subdev, antenna, gain, ext_ref):
        u = self._usrp
        u.set_rx_subdev_spec(uhd.usrp.SubdevSpec(subdev))
        u.set_rx_rate(self._rate)
        u.set_rx_freq(uhd.types.TuneRequest(self._freq))
        u.set_rx_gain(gain)
        u.set_rx_antenna(antenna)
        self._apply_ref(ext_ref)
        print(f"[RX] rate={u.get_rx_rate()/1e6:.3f} Msps  "
              f"freq={u.get_rx_freq()/1e9:.4f} GHz  "
              f"gain={u.get_rx_gain():.1f} dB  ant={u.get_rx_antenna()}")
        # num_recv_frames: number of FPGA→host DMA buffers (bigger = fewer overflows)
        rx_args = uhd.usrp.StreamArgs("fc32", "sc16")
        rx_args.args = "num_recv_frames=512"
        self._streamer = u.get_rx_stream(rx_args)
        cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        cmd.stream_now = True
        self._streamer.issue_stream_cmd(cmd)

    def _apply_ref(self, ext_ref: bool):
        src = "external" if ext_ref else "internal"
        self._usrp.set_clock_source(src)
        self._usrp.set_time_source(src)
        time.sleep(1.0)
        locked = self._usrp.get_mboard_sensor("ref_locked", 0)
        ok = locked.value == "true"
        print(f"  ref={src}  locked={'YES' if ok else 'NO — check OctoClock cable'}")

    # ── Public API ────────────────────────────────────────────────────────────

    def transmit(self, payload: str, msg_id: int = 0) -> Packet:
        """
        Encode payload as a Packet and send one frame + guard silence.
        Returns the Packet that was sent (includes sequence number).

        If the TX pipe breaks (Ethernet hiccup / buffer overflow), the burst
        is terminated cleanly, the stream is reset, and the frame is retried
        once before re-raising.
        """
        pkt = Packet(payload=payload, msg_id=msg_id, seq=self._seq)
        self._seq = (self._seq + 1) & 0xFF

        frame   = _build_frame(pkt)
        silence = np.zeros(SPS * 50, dtype=np.complex64)   # ~400 µs inter-frame gap
        samples = np.concatenate([frame, silence])

        try:
            self._streamer.send(samples, self._md)
        except RuntimeError:
            # Broken pipe — close the burst, wait, then restart and retry once.
            md_eob = uhd.types.TXMetadata()
            md_eob.end_of_burst = True
            try:
                self._streamer.send(np.zeros(100, dtype=np.complex64), md_eob)
            except RuntimeError:
                pass
            time.sleep(0.1)
            # Re-acquire a fresh TX streamer so the USRP is in a clean state.
            self._streamer = self._usrp.get_tx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
            self._md.start_of_burst = True
            self._md.end_of_burst   = False
            self._streamer.send(samples, self._md)   # raises if truly broken

        self._md.start_of_burst = False
        return pkt

    def receive(self) -> tuple[Packet | None, dict]:
        """
        Pull one buffer from the USRP, run preamble detection, decode if found.

        Returns
        -------
        packet    : decoded Packet, or None if nothing valid was found
        info      : dict with keys corr, peak_idx, peak_val, noise_flr
                    (pass to Visualizer.update for live plotting)
        """
        md   = uhd.types.RXMetadata()
        n    = self._streamer.recv(self._recv_buf, md, timeout=3.0)
        info = dict(corr=np.array([]), peak_idx=-1, peak_val=0.0, noise_flr=0.0, power_db=0.0)

        if md.error_code != uhd.types.RXMetadataErrorCode.none:
            # Overflow (O) is harmless — just keep going.
            # Late (L) or anything else: restart the stream so we don't freeze.
            if md.error_code != uhd.types.RXMetadataErrorCode.overflow:
                self._streamer.issue_stream_cmd(
                    uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))
                time.sleep(0.05)
                cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
                cmd.stream_now = True
                self._streamer.issue_stream_cmd(cmd)
            return None, info

        if n < len(BARKER_UP):
            return None, info

        samples  = self._recv_buf[:n]
        # RMS power of the received buffer in dBFS (dB relative to full-scale)
        power_db = 10.0 * np.log10(np.mean(np.abs(samples) ** 2) + 1e-12)

        peak_idx, peak_val, noise_flr, corr = _find_preamble(samples)
        info.update(corr=corr, peak_idx=peak_idx, peak_val=peak_val,
                    noise_flr=noise_flr, power_db=power_db)

        if noise_flr == 0 or peak_val < SNR_THRESHOLD * noise_flr:
            return None, info

        now = time.time()
        if now - self._last_rx < COOLDOWN_S:
            return None, info
        self._last_rx = now

        pre_end = peak_idx + len(BARKER_UP)
        if pre_end > n:
            return None, info
        phase = _estimate_phase(samples[peak_idx:pre_end])

        # Decode the 8-bit frame-length field
        len_end = pre_end + 8 * SPS
        if len_end > n:
            return None, info
        n_bytes = int(np.packbits(_demodulate(samples[pre_end:len_end], 8, phase))[0])

        if n_bytes == 0 or n_bytes > 254:
            return None, info

        msg_end = len_end + n_bytes * 8 * SPS
        if msg_end > n:
            return None, info

        raw = np.packbits(_demodulate(samples[len_end:msg_end], n_bytes * 8, phase)).tobytes()[:n_bytes]

        try:
            return Packet.decode(raw), info
        except (ValueError, UnicodeDecodeError):
            return None, info   # false positive — magic or CRC rejected it

    def close(self):
        if self.role == "tx":
            md = uhd.types.TXMetadata()
            md.end_of_burst = True
            try:
                self._streamer.send(np.zeros(100, dtype=np.complex64), md)
            except RuntimeError:
                pass   # pipe may already be broken; suppress so __exit__ doesn't raise
        else:
            try:
                self._streamer.issue_stream_cmd(
                    uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
                )
            except RuntimeError:
                pass

    def __enter__(self):  return self
    def __exit__(self, *_): self.close()


# ── Visualizer ────────────────────────────────────────────────────────────────

class Visualizer:
    """
    Live two-panel matplotlib figure updated on every receive() call.

    Top panel    : Barker correlation magnitude vs sample index.
                   Red vertical line  = preamble peak.
                   Green dashed line  = noise floor.
                   Orange dotted line = detection threshold (SNR_THRESHOLD × noise).

    Middle panel : Received signal power (dBFS) rolling over time.
                   Each point = RMS power of one receive buffer (~33 ms).
                   Spikes = packet arriving.  Flat = noise floor.

    Bottom panel : Histogram of correlation values — shows separation between
                   the bulk noise distribution and the preamble peak.
    """

    _POWER_HISTORY = 200   # number of buffers to keep in the rolling power plot

    def __init__(self, title: str = "USRP BPSK — Signal Monitor",
                 position: str = "left"):
        """
        Parameters
        ----------
        title    : window title and figure suptitle
        position : "left"  → left half of screen
                   "right" → right half of screen
                   "full"  → full screen (default single-window behaviour)
        """
        if not _MPL:
            raise RuntimeError("matplotlib is not installed — pip install matplotlib")
        from collections import deque
        self._power_history = deque(maxlen=self._POWER_HISTORY)

        plt.ion()
        self.fig = plt.figure(figsize=(12, 8), constrained_layout=True)
        self.fig.canvas.manager.set_window_title(title)

        # Position the window on screen so two plots don't stack
        try:
            mgr = self.fig.canvas.manager
            # Works for TkAgg, Qt5Agg, MacOSX backends
            if hasattr(mgr, "window"):
                w = mgr.window
                # Try to get screen dimensions; fall back to 1920×1080
                try:
                    sw = w.winfo_screenwidth()
                    sh = w.winfo_screenheight()
                except Exception:
                    sw, sh = 1920, 1080
                half_w = sw // 2
                if position == "left":
                    x, y, width, height = 0, 0, half_w, sh
                elif position == "right":
                    x, y, width, height = half_w, 0, half_w, sh
                else:
                    x, y, width, height = 0, 0, sw, sh
                w.geometry(f"{width}x{height}+{x}+{y}")
            elif hasattr(mgr, "canvas") and hasattr(mgr.canvas, "manager"):
                pass  # non-Tk backend — skip geometry, window still opens
        except Exception:
            pass   # positioning is best-effort; never crash the receiver

        gs = self.fig.add_gridspec(3, 1, height_ratios=[2, 1.5, 1.5])
        self.ax_corr  = self.fig.add_subplot(gs[0])
        self.ax_power = self.fig.add_subplot(gs[1])
        self.ax_hist  = self.fig.add_subplot(gs[2])
        self.fig.suptitle(title, fontweight="bold")

        # ── Top: correlation ──────────────────────────────────────────────────
        ax = self.ax_corr
        ax.set_ylabel("|Correlation|")
        (self._ln_corr,) = ax.plot([], [], "b-", lw=0.7, alpha=0.85, label="|corr|")
        self._ln_peak  = ax.axvline(0, color="red",    lw=1.5, label="Peak")
        self._ln_noise = ax.axhline(0, color="green",  lw=1.2, ls="--", label="Noise floor")
        self._ln_thr   = ax.axhline(0, color="orange", lw=1.2, ls=":",  label=f"Threshold (×{SNR_THRESHOLD})")
        ax.set_xticklabels([])
        ax.legend(fontsize=8, loc="upper right")

        # ── Middle: channel power over time ───────────────────────────────────
        ax = self.ax_power
        ax.set_ylabel("Power (dBFS)")
        ax.set_xlabel("Receive buffer (most recent →)")
        (self._ln_power,) = ax.plot([], [], "g-", lw=1.0, label="RMS power")
        ax.legend(fontsize=8, loc="upper right")

        # ── Bottom: histogram ─────────────────────────────────────────────────
        self.ax_hist.set_xlabel("|Correlation|")
        self.ax_hist.set_ylabel("Count")

        self._last_draw = 0.0   # throttle redraws to max 10 Hz

        plt.show(block=False)
        plt.pause(0.05)

    def update(self, info: dict, message: str = ""):
        corr      = info["corr"]
        peak_idx  = info["peak_idx"]
        peak_val  = info["peak_val"]
        noise_flr = info["noise_flr"]
        power_db  = info.get("power_db", None)

        # ── Middle: rolling power — always update, even on UHD errors ────────
        if power_db is not None:
            self._power_history.append(power_db)
        pw = list(self._power_history)
        if pw:
            self._ln_power.set_data(np.arange(len(pw)), pw)
            self.ax_power.relim()
            self.ax_power.autoscale_view()
            self.ax_power.set_xlim(0, self._POWER_HISTORY)
            self.ax_power.set_title(
                f"Channel power: {pw[-1]:.1f} dBFS  "
                f"(min {min(pw):.1f}  max {max(pw):.1f})",
                fontsize=9
            )

        # Redraw power panel even if there is no correlation data this cycle
        if corr.size == 0:
            self._redraw()
            return

        # ── Top: correlation trace ────────────────────────────────────────────
        self._ln_corr.set_data(np.arange(len(corr)), corr)
        self._ln_peak.set_xdata([peak_idx])
        self._ln_noise.set_ydata([noise_flr])
        self._ln_thr.set_ydata([SNR_THRESHOLD * noise_flr])
        self.ax_corr.relim()
        self.ax_corr.autoscale_view()

        snr_str = f"{peak_val / noise_flr:.1f}×" if noise_flr > 0 else "—"
        title   = f"peak={peak_val:.1f}  noise={noise_flr:.1f}  SNR={snr_str}"
        if message:
            title += f'  →  "{message}"'
        self.ax_corr.set_title(title, fontsize=9)

        # ── Bottom: distribution histogram ────────────────────────────────────
        self.ax_hist.clear()
        self.ax_hist.hist(corr, bins=60, color="steelblue", alpha=0.75, label="All samples")
        if peak_idx >= 0:
            self.ax_hist.axvline(peak_val,                color="red",    lw=2,   label=f"Peak  {peak_val:.1f}")
            self.ax_hist.axvline(noise_flr,               color="green",  lw=1.5, ls="--", label=f"Noise  {noise_flr:.1f}")
            self.ax_hist.axvline(SNR_THRESHOLD*noise_flr, color="orange", lw=1.5, ls=":",  label="Threshold")
        self.ax_hist.set_xlabel("|Correlation|")
        self.ax_hist.set_ylabel("Count")
        self.ax_hist.legend(fontsize=8)

        self._redraw()

    def _redraw(self):
        """Flush to screen at most 10 times per second so matplotlib never blocks the RX loop."""
        now = time.monotonic()
        if now - self._last_draw < 0.1:   # 100 ms minimum between redraws
            return
        self._last_draw = now
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
