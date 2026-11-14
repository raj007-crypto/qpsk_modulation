"""
QPSK Project with Audio Input
=============================
Sends an audio clip through a QPSK modulator -> AWGN channel -> coherent
demodulator, rebuilds the audio, and analyses the result.

USAGE
-----
    python qpsk_audio_project.py                    # uses a built-in test melody
    python qpsk_audio_project.py --wav myclip.wav   # uses YOUR clip (mono or stereo)
    python qpsk_audio_project.py --wav myclip.wav --seconds 4

Needs: numpy, scipy, matplotlib      (pip install numpy scipy matplotlib)

OUTPUT (folder  qpsk_audio_results/ )
------
    fig1_audio_original_vs_recovered.png
    fig2_signal_chain_zoom.png
    fig3_constellation.png
    fig4_ber_vs_ebn0.png
    fig5_audio_quality_vs_ebn0.png
    fig6_awgn_waveforms_by_snr.png   <- QPSK carrier + AWGN at 3 chosen Eb/N0 (WAVEFORM_DB)
    original_8bit.wav, recovered_ebn0_XXdB.wav   <- LISTEN TO THESE
    results_table.csv

PIPELINE (matches the project steps)
------------------------------------
 1  Prepare audio       mono, 8 kHz
 2  Quantize            8 bits / sample (unsigned PCM, 0..255)  -> 64 kbit/s
 3  Group bits          2 bits = 1 QPSK symbol
 4  Map to I/Q          00->(+1,+1)  01->(-1,+1)  11->(-1,-1)  10->(+1,-1)
 5  Modulate            s(t) = A [ I cos(2 pi fc t) - Q sin(2 pi fc t) ]
                        (minus sign gives phases 45/135/225/315 deg)
 6  Channel             AWGN at chosen Eb/N0
 7  Demodulate          multiply by cos / -sin, integrate over Ts, take sign
 8  Rebuild audio       bit pairs -> bytes -> amplitude
 9  Plots & metrics
Time is normalised (Ts = 1); the carrier uses FC cycles per symbol, so no real
RF frequency is needed - the maths is identical.
"""
import argparse
import csv
import os
from fractions import Fraction

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample_poly
from scipy.special import erfc

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
FS_AUDIO = 8000            # audio sample rate (Hz)
SPS = 16                   # samples per QPSK symbol in the simulation
FC = 2                     # carrier cycles per symbol
TS = 1.0                   # symbol duration (normalised)
EB = 1.0                   # energy per bit (normalised)
ES = 2 * EB                # energy per symbol
A = np.sqrt(ES / TS)       # carrier amplitude factor
SWEEP_DB = [0, 2, 4, 6, 8, 10, 12]     # Eb/N0 values tested on the audio
DEMO_DB = [0, 4, 8, 12]                # values shown in the audio plot (Fig. 1)
WAVEFORM_DB = [0, 4, 12]               # Eb/N0 values drawn in the QPSK+AWGN waveform plot (Fig. 6)
OUT_DIR = "qpsk_audio_results"
rng = np.random.default_rng(1)         # fixed seed -> repeatable results


# ----------------------------------------------------------------------------
# Step 1 : prepare audio
# ----------------------------------------------------------------------------
def make_test_audio(seconds=3.0, fs=FS_AUDIO):
    """Synthetic 'melody' (notes with harmonics + envelope) - used if no WAV given."""
    notes = [262, 330, 392, 523, 392, 330, 294, 392, 494, 392, 262, 262]   # Hz
    per = seconds / len(notes)
    x = []
    for f in notes:
        t = np.arange(int(per * fs)) / fs
        env = np.minimum(1, t / 0.02) * np.exp(-2.5 * t)                 # attack + decay
        vib = 1 + 0.004 * np.sin(2 * np.pi * 5 * t)                      # slight vibrato
        tone = (np.sin(2 * np.pi * f * vib * t)
                + 0.5 * np.sin(2 * np.pi * 2 * f * t)
                + 0.25 * np.sin(2 * np.pi * 3 * f * t))
        x.append(env * tone)
    x = np.concatenate(x)
    return x / np.max(np.abs(x)) * 0.9


def load_audio(path, seconds):
    sr, raw = wavfile.read(path)
    data = raw.astype(np.float64)
    if np.issubdtype(raw.dtype, np.integer):          # scale integer PCM to [-1, 1]
        data = (data - 128) / 128 if raw.dtype == np.uint8 else data / (np.iinfo(raw.dtype).max + 1)
    if data.ndim > 1:                                 # stereo -> mono
        data = data.mean(axis=1)
    data = data[: int(seconds * sr)]
    frac = Fraction(FS_AUDIO, sr).limit_denominator(1000)
    data = resample_poly(data, frac.numerator, frac.denominator)   # -> 8 kHz
    return data / (np.max(np.abs(data)) + 1e-12) * 0.9


# ----------------------------------------------------------------------------
# Step 2 : quantize  /  Step 8 : de-quantize
# ----------------------------------------------------------------------------
def quantize_8bit(x):
    return np.clip(np.round((x + 1) / 2 * 255), 0, 255).astype(np.uint8)


def dequantize_8bit(u8):
    return u8.astype(np.float64) / 255 * 2 - 1


def bytes_to_bits(u8):
    return np.unpackbits(u8)                      # MSB first


def bits_to_bytes(bits):
    return np.packbits(bits.astype(np.uint8))


# ----------------------------------------------------------------------------
# Steps 3-4 : bit pairs -> I/Q
# ----------------------------------------------------------------------------
def bits_to_iq(bits):
    """00->(+1,+1) 01->(-1,+1) 11->(-1,-1) 10->(+1,-1)   (Gray)"""
    bits = np.asarray(bits).astype(int)      # int avoids uint8 wrap-around
    b0, b1 = bits[0::2], bits[1::2]
    I = 1 - 2 * b1          # 2nd bit decides I
    Q = 1 - 2 * b0          # 1st bit decides Q
    return I.astype(float), Q.astype(float)


def iq_to_bits(I, Q):
    bits = np.empty(2 * len(I), dtype=np.uint8)
    bits[0::2] = (Q < 0)
    bits[1::2] = (I < 0)
    return bits


# ----------------------------------------------------------------------------
# Step 5 : modulation
# ----------------------------------------------------------------------------
def modulate(I, Q, sps=SPS, fc=FC):
    t = np.arange(sps) / sps * TS
    c = np.cos(2 * np.pi * fc * t)
    s = np.sin(2 * np.pi * fc * t)
    return A * (I[:, None] * c[None, :] - Q[:, None] * s[None, :])     # (Nsym, sps)


# ----------------------------------------------------------------------------
# Step 6 : AWGN channel
# ----------------------------------------------------------------------------
def awgn(s, ebn0_db, sps=SPS):
    if ebn0_db is None:                       # noise-free (for testing)
        return s.copy()
    N0 = EB / (10 ** (ebn0_db / 10))
    sigma = np.sqrt(N0 / 2 * (sps / TS))      # sampled-noise std for PSD N0/2
    return s + sigma * rng.standard_normal(s.shape)


# ----------------------------------------------------------------------------
# Step 7 : coherent demodulation (correlator = multiply + integrate over Ts)
# ----------------------------------------------------------------------------
def demodulate(r, sps=SPS, fc=FC):
    t = np.arange(sps) / sps * TS
    c = np.cos(2 * np.pi * fc * t)
    s = np.sin(2 * np.pi * fc * t)
    fs = sps / TS
    k = 2 / (A * TS) / fs
    I_hat = (r @ c) * k                       # ~ +-1 plus noise
    Q_hat = -(r @ s) * k
    return I_hat, Q_hat, iq_to_bits(I_hat, Q_hat)


# ----------------------------------------------------------------------------
# Full chain for one Eb/N0
# ----------------------------------------------------------------------------
def run_chain(bits_tx, ebn0_db):
    I, Q = bits_to_iq(bits_tx)
    r = awgn(modulate(I, Q), ebn0_db)
    I_hat, Q_hat, bits_rx = demodulate(r)
    return dict(bits_rx=bits_rx, I_hat=I_hat, Q_hat=Q_hat,
                ber=float(np.mean(bits_rx != bits_tx)),
                n_err=int(np.sum(bits_rx != bits_tx)))


def audio_snr_db(ref, test):
    err = np.sum((ref - test) ** 2)
    return np.inf if err == 0 else 10 * np.log10(np.sum(ref ** 2) / err)


def theory_ber(ebn0_db):
    return 0.5 * erfc(np.sqrt(10 ** (np.asarray(ebn0_db, float) / 10)))


def ber_random_baseband(ebn0_db, n_bits):
    """Fast baseband-equivalent QPSK on random bits (for the long BER curve)."""
    bits = rng.integers(0, 2, n_bits).astype(np.uint8)
    I, Q = bits_to_iq(bits)
    sigma = np.sqrt(EB / (10 ** (ebn0_db / 10)) / 2)
    I = I + sigma * rng.standard_normal(len(I))
    Q = Q + sigma * rng.standard_normal(len(Q))
    return float(np.mean(iq_to_bits(I, Q) != bits))


# ----------------------------------------------------------------------------
# Test the chain on a few samples first (as the project plan suggests)
# ----------------------------------------------------------------------------
def self_test():
    u8 = np.array([0, 1, 127, 128, 200, 255, 37, 90], dtype=np.uint8)
    bits = bytes_to_bits(u8)
    out = run_chain(bits, None)                         # no noise
    assert np.array_equal(out["bits_rx"], bits), "noise-free bits do not match!"
    assert np.array_equal(bits_to_bytes(out["bits_rx"]), u8), "byte rebuild failed!"
    # check the carrier phases are 45/135/225/315 deg
    for pair, deg in {(0, 0): 45, (0, 1): 135, (1, 1): 225, (1, 0): 315}.items():
        I, Q = bits_to_iq(np.array(pair, dtype=np.uint8))
        ph = np.degrees(np.arctan2(Q, I))[0] % 360
        assert abs(ph - deg) < 1e-9, f"phase mismatch for {pair}"
    s = modulate(*bits_to_iq(np.array([0, 0], dtype=np.uint8)))[0]
    t = np.arange(SPS) / SPS
    assert np.allclose(s, np.sqrt(2) * A * np.cos(2 * np.pi * FC * t + np.pi / 4)), \
        "carrier phase for 00 is not 45 deg"
    print("[self-test] PASSED: noise-free recovery is exact, phases are 45/135/225/315 deg\n")


# ============================================================================
# Plots
# ============================================================================
def step(ax, x, y, width=1, **kw):
    """Draw a staircase; each value lasts `width` time units."""
    ax.step(np.r_[x, x[-1] + width], np.r_[y, y[-1]], where="post", **kw)


def fig_audio(x_orig, ref, recs, t):
    fig, ax = plt.subplots(1 + len(recs), 1, figsize=(10, 2.3 * (1 + len(recs))),
                           sharex=True, constrained_layout=True)
    ax[0].plot(t, x_orig, lw=0.7, color="tab:blue")
    ax[0].set_title("Original audio (8 kHz)")
    for a, (db, rec, snr, ber) in zip(ax[1:], recs):
        a.plot(t, rec, lw=0.7, color="tab:red")
        a.set_title(f"Recovered audio at Eb/N0 = {db} dB   (BER = {ber:.2e}, "
                    f"audio SNR = {snr:.1f} dB)" if np.isfinite(snr) else
                    f"Recovered audio at Eb/N0 = {db} dB   (BER = {ber:.2e}, error-free)")
    for a in ax:
        a.set_ylim(-1.05, 1.05); a.grid(True, ls="--", alpha=0.4); a.set_ylabel("Amplitude")
    ax[-1].set_xlabel("Time (s)")
    fig.savefig(f"{OUT_DIR}/fig1_audio_original_vs_recovered.png", dpi=170)
    plt.close(fig)


def fig_chain(bits, I, Q, ebn0_db=8, n_sym=8, sps_plot=64, start_sym=17600):
    """Zoom on n_sym symbols starting at symbol `start_sym` (~0.55 s into the clip)."""
    start_sym = min(start_sym, max(0, len(I) - n_sym))
    nb = 2 * n_sym
    bits = bits[2 * start_sym: 2 * start_sym + nb]
    Ii, Qi = I[start_sym:start_sym + n_sym], Q[start_sym:start_sym + n_sym]
    s = modulate(Ii, Qi, sps=sps_plot)
    # noisy version at plotting resolution
    N0 = EB / (10 ** (ebn0_db / 10))
    r = s + np.sqrt(N0 / 2 * sps_plot) * rng.standard_normal(s.shape)
    t = np.arange(n_sym * sps_plot) / sps_plot * 2          # 1 symbol = 2 bit periods

    fig, ax = plt.subplots(5, 1, figsize=(10, 11), constrained_layout=True)
    step(ax[0], np.arange(nb), bits[:nb], color="tab:blue", lw=2)
    ax[0].set_title(f"Input bit stream (zoom: {nb} bits taken from the middle of the audio data)")
    ax[0].set_yticks([0, 1])
    for k in range(n_sym):
        ax[0].text(2 * k + 1, 1.15, f"{bits[2*k]}{bits[2*k+1]}", ha="center", fontsize=9)
    ax[0].set_ylim(-0.3, 1.5)

    step(ax[1], 2 * np.arange(n_sym), Ii, width=2, color="green", lw=2)
    ax[1].set_title("In-phase (I) waveform"); ax[1].set_yticks([-1, 1])
    step(ax[2], 2 * np.arange(n_sym), Qi, width=2, color="darkorange", lw=2)
    ax[2].set_title("Quadrature (Q) waveform"); ax[2].set_yticks([-1, 1])
    ax[3].plot(t, s.ravel(), color="red", lw=1.2)
    ax[3].set_title("QPSK modulated carrier (zoom) - phase jumps at symbol boundaries")
    ax[4].plot(t, r.ravel(), color="gray", lw=0.8)
    ax[4].set_title(f"Received signal after AWGN (Eb/N0 = {ebn0_db} dB)")
    ax[4].set_xlabel("Time (bit periods)")
    for a in ax:
        a.set_xlim(0, nb); a.grid(True, ls="--", alpha=0.4)
        for k in range(0, nb + 1, 2):
            a.axvline(k, color="k", lw=0.4, alpha=0.4)
    fig.savefig(f"{OUT_DIR}/fig2_signal_chain_zoom.png", dpi=170)
    plt.close(fig)


def fig_constellation(clean, noisy):
    panels = [("Clean (no noise)", clean)] + [(f"Eb/N0 = {db} dB", d) for db, d in noisy]
    fig, axs = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.9), constrained_layout=True)
    ideal = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    for ax, (title, d) in zip(axs, panels):
        n = min(4000, len(d["I_hat"]))
        ax.scatter(d["I_hat"][:n], d["Q_hat"][:n], s=3, alpha=0.3, color="tab:blue")
        ax.scatter(ideal[:, 0], ideal[:, 1], s=70, color="red", edgecolor="k", zorder=5)
        ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
        ax.set_title(title); ax.set_xlabel("I"); ax.set_ylabel("Q")
        ax.set_aspect("equal"); ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
        ax.grid(True, ls="--", alpha=0.4)
    fig.suptitle("QPSK constellation of the audio data (after demodulator)", fontsize=13)
    fig.savefig(f"{OUT_DIR}/fig3_constellation.png", dpi=170)
    plt.close(fig)


def fig_ber(audio_pts):
    ebn0 = np.arange(0, 10.5, 1.0)
    sim = []
    for db in ebn0:
        n = int(min(1e7, max(2e5, 150 / theory_ber(db))))
        sim.append(ber_random_baseband(db, n + n % 2))
    x = np.linspace(0, 10, 200)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.semilogy(x, theory_ber(x), "k-", lw=2, label=r"Theory: $\frac{1}{2}\mathrm{erfc}(\sqrt{E_b/N_0})$")
    ax.semilogy(ebn0, sim, "o", mfc="none", ms=7, color="tab:blue", label="Simulation: random bits")
    d = [(db, b) for db, b in audio_pts if b > 0]
    if d:
        ax.semilogy(*zip(*d), "s", ms=6, color="tab:red", label="Simulation: audio bits (full chain)")
    ax.set_xlabel(r"$E_b/N_0$ (dB)"); ax.set_ylabel("BER")
    ax.set_title("QPSK BER over AWGN: simulation vs theory")
    ax.set_ylim(1e-6, 1); ax.grid(True, which="both", ls="--", alpha=0.4); ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig4_ber_vs_ebn0.png", dpi=170)
    plt.close(fig)


def fig_quality(rows):
    db = [r["ebn0"] for r in rows]
    snr = [min(r["snr"], 60) for r in rows]          # cap "infinite" at 60 for plotting
    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    ax1.plot(db, snr, "o-", color="tab:red")
    ax1.set_xlabel(r"$E_b/N_0$ (dB)"); ax1.set_ylabel("Recovered-audio SNR (dB)", color="tab:red")
    ax1.grid(True, ls="--", alpha=0.4)
    ax2 = ax1.twinx()
    ber = [max(r["ber"], 1e-7) for r in rows]        # floor so zero errors can be drawn
    ax2.semilogy(db, ber, "s--", color="tab:blue")
    ax2.set_ylabel("Measured BER (floor 1e-7 = error-free)", color="tab:blue")
    ax1.set_title("Audio quality vs channel quality\n(error-free recovery has infinite SNR vs the 8-bit original; drawn at 60 dB)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig5_audio_quality_vs_ebn0.png", dpi=170)
    plt.close(fig)


def fig_awgn_waveforms(bits, I, Q, levels=WAVEFORM_DB, n_sym=8, sps_plot=64, start_sym=17600):
    """QPSK carrier after the AWGN channel, one row per Eb/N0, same 8 symbols in every row.
    The clean carrier is drawn in red underneath so the effect of the noise is visible.
    Each row is also demodulated, so you can see the decoded result for that window."""
    start_sym = min(start_sym, max(0, len(I) - n_sym))
    Ii, Qi = I[start_sym:start_sym + n_sym], Q[start_sym:start_sym + n_sym]
    tx_bits = bits[2 * start_sym: 2 * start_sym + 2 * n_sym]
    s = modulate(Ii, Qi, sps=sps_plot)
    t = np.arange(n_sym * sps_plot) / sps_plot * 2            # 1 symbol = 2 bit periods
    nb = 2 * n_sym

    fig, ax = plt.subplots(1 + len(levels), 1, figsize=(10, 1.9 * (1 + len(levels))),
                           sharex=True, constrained_layout=True)
    ax[0].plot(t, s.ravel(), color="red", lw=1.3)
    ax[0].set_title("Transmitted QPSK carrier (no noise)", fontsize=10)

    for a, db in zip(ax[1:], levels):
        N0 = EB / (10 ** (db / 10))
        r = s + np.sqrt(N0 / 2 * sps_plot) * rng.standard_normal(s.shape)
        _, _, rx_bits = demodulate(r, sps=sps_plot)                # decode this window
        n_err = int(np.sum(rx_bits != tx_bits))
        a.plot(t, r.ravel(), color="gray", lw=0.7, label="received (with AWGN)")
        a.plot(t, s.ravel(), color="red", lw=1.0, alpha=0.8, label="clean carrier")
        a.set_title(f"Eb/N0 = {db} dB  -  bit errors in this window: {n_err} / {nb}", fontsize=10)
    ax[1].legend(loc="upper right", fontsize=8, ncol=2)

    for a in ax:
        a.set_xlim(0, nb); a.grid(True, ls="--", alpha=0.4); a.set_ylabel("Amplitude")
        for k in range(0, nb + 1, 2):
            a.axvline(k, color="k", lw=0.4, alpha=0.4)
    ax[-1].set_xlabel("Time (bit periods; dotted lines = symbol boundaries)")
    fig.suptitle("QPSK carrier after the AWGN channel at different Eb/N0\n"
                 "(the noise looks large because it is wideband; the receiver integrates over each "
                 "symbol, which averages most of it out)", fontsize=11)
    fig.savefig(f"{OUT_DIR}/fig6_awgn_waveforms_by_snr.png", dpi=170)
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", help="path to your own WAV clip (optional)")
    ap.add_argument("--seconds", type=float, default=3.0, help="clip length to use (2-5 s)")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    self_test()

    # --- Steps 1-2 -----------------------------------------------------------
    x = load_audio(args.wav, args.seconds) if args.wav else make_test_audio(args.seconds)
    u8 = quantize_8bit(x)
    ref = dequantize_8bit(u8)                       # what a perfect channel would give
    bits = bytes_to_bits(u8)
    t = np.arange(len(x)) / FS_AUDIO
    q_snr = audio_snr_db(x, ref)
    print(f"Audio: {len(x)} samples = {len(x)/FS_AUDIO:.2f} s | bits = {len(bits)} "
          f"({len(bits)//2} QPSK symbols) | 8-bit quantisation SNR = {q_snr:.1f} dB")
    wavfile.write(f"{OUT_DIR}/original_8bit.wav", FS_AUDIO, (ref * 32767).astype(np.int16))

    # --- Steps 3-8 for every Eb/N0 --------------------------------------------
    rows, store = [], {}
    print("\nEb/N0(dB) | bit errors |    BER    | audio SNR (dB)")
    for db in SWEEP_DB:
        out = run_chain(bits, db)
        rec = dequantize_8bit(bits_to_bytes(out["bits_rx"]))
        snr = audio_snr_db(ref, rec)
        store[db] = (out, rec, snr)
        rows.append(dict(ebn0=db, n_err=out["n_err"], ber=out["ber"], theory=float(theory_ber(db)), snr=snr))
        wavfile.write(f"{OUT_DIR}/recovered_ebn0_{db:02d}dB.wav", FS_AUDIO, (rec * 32767).astype(np.int16))
        print(f"{db:8d}  | {out['n_err']:10d} | {out['ber']:.3e} | {snr:8.2f}" if np.isfinite(snr)
              else f"{db:8d}  | {out['n_err']:10d} | {out['ber']:.3e} |  error-free")

    with open(f"{OUT_DIR}/results_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EbN0_dB", "bit_errors", "BER_simulated", "BER_theory", "audio_SNR_dB"])
        for r in rows:
            w.writerow([r["ebn0"], r["n_err"], f"{r['ber']:.3e}", f"{r['theory']:.3e}",
                        "inf" if not np.isfinite(r["snr"]) else f"{r['snr']:.2f}"])

    # --- Plots -----------------------------------------------------------------
    I, Q = bits_to_iq(bits)
    recs = [(db, store[db][1], store[db][2], store[db][0]["ber"]) for db in DEMO_DB]
    fig_audio(x, ref, recs, t)
    fig_chain(bits, I, Q)
    clean = run_chain(bits, None)
    fig_constellation(clean, [(db, store[db][0]) for db in [8, 4, 0]])
    fig_ber([(r["ebn0"], r["ber"]) for r in rows])
    fig_quality(rows)
    fig_awgn_waveforms(bits, I, Q)
    print(f"\nDone. Everything is saved in the folder '{OUT_DIR}/'.")
    print("Listen to original_8bit.wav and the recovered_ebn0_XXdB.wav files to hear the effect of noise.")


if __name__ == "__main__":
    main()