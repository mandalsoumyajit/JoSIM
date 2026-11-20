"""The fork's solver paths, output formats and analysis modes, checked against each other.

Every path must agree with the plain MNA loop: the nodal solver (-n), the OpenMP options (-p, -p -n; serial when
OpenMP is off), both analysis types, events-only output, print-grid decimation, Monte Carlo and parameter sweeps,
saved and loaded states, and every output format. Bad input must fail with the matching diagnostic.
Usage: regression_fork_modes.py <josim-cli>
"""

import ast
import csv
import math
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest


BINARY = pathlib.Path(sys.argv.pop(1)).resolve()
TESTS = pathlib.Path(__file__).resolve().parent
DFF = TESTS / "ex_mitll_dff_wr.cir"
JTL = TESTS / "ex_jtl_basic.cir"
XOR = TESTS / "ex_mitll_xor_wr.cir"
SINK = TESTS / "ex_dcsfq_jtl_sink.cir"

# A short JTL with a transmission line and a spread, for Monte Carlo, sweeps, decimation and state files.
JTL_DECK = """* JTL with a line
.model jmod jj(rtype=1, vg=2.8mV, cap=0.07pF, r0=160ohm, rn=16ohm, icrit=0.25mA)
{spread}
.param BIAS=175u
IB1 0 1 {bias}
IB2 0 3 {bias}
IB3 0 5 {bias}
IIN 0 1 {pulse}
B1 1 0 jmod
RS1 1 0 3
L1 1 2 2p
T1 2 0 3 0 TD=3p Z0=2
B2 3 0 jmod
RS2 3 0 3
L2 3 5 2p
B3 5 0 jmod
RS3 5 0 3
L3 5 6 2p
RL 6 0 5
.tran 0.25p {tstop} {prstart} {prstep}
.print p(B1) p(B2) p(B3) i(L3)
{extra}
.end
"""


PULSES = "pwl(0 0 30p 0 32p 500u 34p 0 70p 0 72p 500u 74p 0)"

# Every linear device kind on both sides of a transmission line (partitions), with a spread for Monte Carlo.
RICH = """* rich deck
.model jmod jj(rtype=1, vg=2.8mV, cap=0.07pF, r0=160ohm, rn=16ohm, icrit=0.25mA)
.spread 0
IB1 0 1 pwl(0 0 10p 175u)
IB2 0 4 pwl(0 0 10p 175u)
IIN 0 1 pwl(0 0 30p 0 32p 300u 34p 0 60p 0 62p 300u 64p 0)
B1 1 0 jmod
RS1 1 0 3
L1 1 2 2p
LK 7 0 1p
K1 L1 LK 0.1
RK 7 0 1
C1 2 0 0.05p
T1 2 0 3 0 TD=3p Z0=2
L2 3 4 2p
B2 4 0 jmod
RS2 4 0 3
V1 5 0 pwl(0 0 40p 0 41p 1m 42p 0)
R5 5 4 50
P1 6 0 pwl(0 0 20p 0 30p 3.14159)
R6 6 4 100
G1 0 4 5 0 0.001
E1 8 0 4 0 2
R8 8 0 10
H1 9 0 5 4 5
R9 9 0 10
RL 4 0 5
.tran 0.25p 100p 0 0.25p
.print p(B1) p(B2) v(4) i(L2) v(8)
.end
"""


def jtl(spread="", tstop="120p", prstart="0", prstep="0.25p", extra="", bias="pwl(0 0 10p BIAS)", pulse=PULSES):
    return JTL_DECK.format(spread=spread, tstop=tstop, prstart=prstart, prstep=prstep, extra=extra, bias=bias,
                           pulse=pulse)


def read_csv(path):
    with open(path, newline="") as stream:
        rows = list(csv.reader(stream))
    names = [name.strip('"') for name in rows[0]]
    return names, [[float(value) for value in row] for row in rows[1:]]


def read_npy(path):
    """A JoSIM .npy structured array (little-endian f8/i4/S fields) as (names, rows)."""
    data = pathlib.Path(path).read_bytes()
    if data[:6] != b"\x93NUMPY":
        raise AssertionError("not an .npy file")
    major = data[6]
    if major == 1:
        (length,), start = struct.unpack_from("<H", data, 8), 10
    else:
        (length,), start = struct.unpack_from("<I", data, 8), 12
    header = ast.literal_eval(data[start:start + length].decode("latin1"))
    fields = header["descr"]
    count = header["shape"][0]
    fmt, names = "<", []
    for name, kind in fields:
        names.append(name.strip('"'))
        if kind == "<f8":
            fmt += "d"
        elif kind == "<i4":
            fmt += "i"
        elif kind.startswith("|S"):
            fmt += kind[2:] + "s"
        else:
            raise AssertionError("unexpected field type " + kind)
    size = struct.calcsize(fmt)
    offset = start + length
    rows = []
    for index in range(count):
        values = struct.unpack_from(fmt, data, offset + index * size)
        rows.append([value.rstrip(b"\0").decode() if isinstance(value, bytes) else value for value in values])
    return names, rows


def read_bin(path):
    """A JOSIMBIN v1 file (column-major float64) as (names, rows)."""
    data = pathlib.Path(path).read_bytes()
    if data[:8] != b"JOSIMBIN":
        raise AssertionError("not a JOSIMBIN file")
    version, _flags, columns, count = struct.unpack_from("<IIQQ", data, 8)
    if version != 1:
        raise AssertionError("JOSIMBIN version")
    offset, names = 32, []
    for _ in range(columns):
        (length,) = struct.unpack_from("<I", data, offset)
        names.append(data[offset + 4:offset + 4 + length].decode().strip('"'))
        offset += 4 + length + 1
    values = struct.unpack_from("<%dd" % (columns * count), data, offset)
    return names, [[values[c * count + r] for c in range(columns)] for r in range(count)]


def read_raw(path):
    """SPICE raw output, ASCII or binary (float64), as (number of variables, rows)."""
    data = pathlib.Path(path).read_bytes()
    variables = points = None
    for line in data.split(b"\n"):
        if line.startswith(b"No. Variables:"):
            variables = int(line.split(b":")[1])
        elif line.startswith(b"No. Points:"):
            points = int(line.split(b":")[1])
        if line.startswith(b"Binary:") or line.startswith(b"Values:"):
            break
    if b"Binary:\n" in data:
        offset = data.index(b"Binary:\n") + len(b"Binary:\n")
        values = struct.unpack_from("<%dd" % (variables * points), data, offset)
        return variables, [list(values[r * variables:(r + 1) * variables]) for r in range(points)]
    tokens = data[data.index(b"Values:") + len(b"Values:"):].split()
    rows, index = [], 0
    for _ in range(points):
        index += 1   # point number
        rows.append([float(tokens[index + k]) for k in range(variables)])
        index += variables
    return variables, rows


def crossings(times, phases):
    """Times at which a phase trace passes pi + 2 pi k upwards (linear interpolation), the fork's SFQ events."""
    events = []
    for i in range(1, len(times)):
        lower, upper = phases[i - 1], phases[i]
        if upper <= lower:
            continue
        k = math.floor((lower - math.pi) / (2 * math.pi)) + 1
        while math.pi + 2 * math.pi * k <= upper:
            level = math.pi + 2 * math.pi * k
            if level > lower:
                events.append(times[i - 1] + (times[i] - times[i - 1]) * (level - lower) / (upper - lower))
            k += 1
    return events


def phase_events(names, rows):
    times = [row[0] for row in rows]
    return {name: crossings(times, [row[i] for row in rows])
            for i, name in enumerate(names) if name.startswith("P(")}


class ForkModes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="fork-modes-", dir=pathlib.Path.cwd())
        self.dir = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_josim(self, args, deck=None, success=True, stdin=None, timeout=600):
        if isinstance(deck, str):
            path = self.dir / "input.cir"
            path.write_text(deck)
            deck = path
        command = [str(BINARY), *args] + ([str(deck)] if deck is not None else [])
        # A fixed seed: Monte Carlo runs on different paths draw the same trials.
        environment = dict(os.environ, JOSIM_SEED="7")
        result = subprocess.run(command, cwd=self.dir, capture_output=True, text=True, input=stdin,
                                timeout=timeout, env=environment)
        output = result.stdout + result.stderr
        if success:
            self.assertEqual(result.returncode, 0, " ".join(command) + "\n" + output[-3000:])
        else:
            self.assertNotEqual(result.returncode, 0, " ".join(command) + "\n" + output[-3000:])
        return output

    def waveform(self, args, deck, name="out.csv"):
        self.run_josim(["-m", "1", *args, "-o", name], deck)
        return read_csv(self.dir / name)

    def assert_events_match(self, reference, other, tolerance, label):
        self.assertEqual(sorted(reference), sorted(other), label)
        for trace, times in reference.items():
            self.assertEqual(len(times), len(other[trace]), f"{label}: {trace} event count")
            for a, b in zip(times, other[trace]):
                self.assertLess(abs(a - b), tolerance, f"{label}: {trace} event at {a:.6e}")

    # ---- solver paths ----
    def events(self, args, deck):
        """The switching events (trace -> upward event times) a run reports with -e."""
        self.run_josim(["-m", "1", *args, "-e", "ev.csv"], deck)
        found = {}
        for trace, time, polarity in self.read_events_csv():
            found.setdefault(trace, [])
            if polarity > 0:
                found[trace].append(time)
        return found

    def test_solver_paths_agree(self):
        for deck in (DFF, JTL, XOR, SINK, jtl()):
            label = deck.name if isinstance(deck, pathlib.Path) else "jtl"
            with self.subTest(deck=label):
                reference = self.events(["-a", "1"], deck)
                self.assertTrue(any(reference.values()), "deck has no switching events")
                for args in (["-n", "-a", "1"], ["-p", "-a", "1"], ["-p", "-n", "-a", "1"]):
                    self.assert_events_match(reference, self.events(args, deck), 1e-13, f"{label} {' '.join(args)}")
                # Voltage analysis extracts events of device phases only (a node phase such as P(3,0) gives none).
                if any("," in trace for trace in reference):
                    continue
                for args in (["-a", "0"], ["-n", "-a", "0"]):
                    self.assert_events_match(reference, self.events(args, deck), 0.2e-12, f"{label} {' '.join(args)}")

    def test_corrector_and_fixed_step(self):
        reference = self.events([], jtl())
        for args in (["-c", "2"], ["-n", "-c"], ["-f"], ["-n", "-f"], ["-n", "-c", "3", "-f"]):
            self.assert_events_match(reference, self.events(args, jtl()), 0.2e-12, " ".join(args))

    def test_coarse_step_rolls_back_locally(self):
        deck = jtl(prstep="2p").replace(".tran 0.25p", ".tran 2p")
        for args in ([], ["-n"]):
            output = self.run_josim(["-V", "1", *args, "-o", "coarse.csv"], deck)
            self.assertTrue("rollback" in output.lower() or "halving" in output.lower(), output[-2000:])
            names, rows = read_csv(self.dir / "coarse.csv")
            self.assertGreater(len(rows), 10)

    # ---- output formats ----
    def test_output_formats_carry_the_same_data(self):
        names, rows = self.waveform([], JTL)
        # CSV has 7 significant digits; values near zero (residual currents) differ in their last bits between runs
        # on some platforms (arm64), so the absolute part of the tolerance scales with each column's range.
        scale = [max(abs(row[c]) for row in rows) or 1.0 for c in range(len(names))]
        for extension, reader in (("dat", None), ("npy", read_npy), ("bin", read_bin)):
            self.run_josim(["-m", "1", "-o", "out." + extension], JTL)
            if reader is None:
                with open(self.dir / "out.dat") as stream:
                    lines = stream.read().split("\n")
                other = [[float(v) for v in line.split()] for line in lines[1:] if line.strip()]
                other_names = [n.strip('"') for n in lines[0].split()]
            else:
                other_names, other = reader(self.dir / ("out." + extension))
            self.assertEqual(other_names, names, extension)
            self.assertEqual(len(other), len(rows), extension)
            for a, b in zip(rows, other):
                for c, (x, y) in enumerate(zip(a, b)):
                    self.assertLessEqual(abs(x - y), 1e-6 * abs(x) + 1e-12 * scale[c], extension)
        for extension in ("raw", "rawb"):
            self.run_josim(["-m", "1", "-o", "out." + extension], JTL)
            variables, other = read_raw(self.dir / ("out." + extension))
            self.assertEqual(variables, len(names), extension)
            self.assertEqual(len(other), len(rows), extension)
            for a, b in zip(rows, other):
                for c, (x, y) in enumerate(zip(a, b)):
                    self.assertLessEqual(abs(x - y), 5e-6 * abs(x) + 1e-12 * scale[c], extension)

    def test_voltage_mode_formats(self):
        for extension, reader in (("npy", read_npy), ("bin", read_bin)):
            self.run_josim(["-m", "1", "-n", "-a", "0", "-o", "v." + extension], DFF)
            names, rows = reader(self.dir / ("v." + extension))
            self.assertEqual(names[0], "time")
            self.assertGreater(len(rows), 100)

    # ---- events ----
    def test_events_only_matches_waveforms(self):
        for deck in (DFF, JTL, XOR):
            with self.subTest(deck=deck.name):
                names, rows = self.waveform(["-n"], deck)
                expected = phase_events(names, rows)
                for args in (["-n", "-e", "ev.csv"], ["-e", "ev.csv"], ["-n", "-o", "w.csv", "-e", "ev.csv"]):
                    self.run_josim(["-m", "1", *args], deck)
                    events = self.read_events_csv()
                    got = {}
                    for trace, time, polarity in events:
                        if polarity > 0:
                            got.setdefault(trace, []).append(time)
                    for trace in expected:
                        self.assertIn(trace, got if expected[trace] else {trace: []}, " ".join(args))
                        self.assertEqual(len(got.get(trace, [])), len(expected[trace]), f"{trace} {args}")
                        for a, b in zip(sorted(got.get(trace, [])), expected[trace]):
                            self.assertLess(abs(a - b), 0.3e-12, f"{trace} {args}")
                self.run_josim(["-m", "1", "-n", "-e", "ev.npy"], deck)
                npy_names, npy_rows = read_npy(self.dir / "ev.npy")
                self.assertEqual(npy_names[:6], ["trace_id", "trace", "time", "polarity", "winding", "phase"])
                csv_events = self.read_events_csv()
                self.assertEqual(len(npy_rows), len(csv_events))

    def read_events_csv(self):
        with open(self.dir / "ev.csv", newline="") as stream:
            rows = list(csv.reader(stream))
        header = rows[0]
        tail = len(header) - header.index("time")   # fields from time onwards
        events = []
        for row in rows[1:]:
            extra = len(row) - len(header)          # commas inside an unquoted trace name
            trace = ",".join(row[1:2 + extra]).strip('"')
            fields = row[2 + extra:]
            events.append((trace, float(fields[0]), int(fields[1])))
            self.assertEqual(len(fields), tail)
        return events

    def test_event_metrics(self):
        for metrics in ("all", "pulse", "t_peak,v_peak,time=cross"):
            self.run_josim(["-m", "1", "-n", "-e", "ev.csv", "-E", metrics], JTL)
            with open(self.dir / "ev.csv", newline="") as stream:
                rows = list(csv.reader(stream))
            header = rows[0]
            self.assertIn("t_peak", header, metrics)
            peak, time = header.index("t_peak"), header.index("time")
            for row in rows[1:]:
                self.assertLess(abs(float(row[peak]) - float(row[time])), 5e-12, metrics)

    def test_golden_event_verification(self):
        self.run_josim(["-m", "1", "-e", "golden.csv"], DFF)
        output = self.run_josim(["--verify-events=golden.csv"], DFF)
        self.assertIn("Event Verification: PASS", output)
        self.run_josim(["-m", "1", "-n", "-e", "golden.npy"], DFF)
        output = self.run_josim(["-n", "--verify-events=golden.npy", "-T", "1p"], DFF)
        self.assertIn("PASS", output)

    # ---- print-grid decimation ----
    def test_decimated_output_equals_full_run(self):
        for args in ([], ["-n"]):
            full_names, full = self.run_npy(args, jtl())
            index = {round(row[0] * 1e16): row for row in full}
            for prstart, prstep in (("20p", "1p"), ("0", "2.5p"), ("50p", "0.25p")):
                names, rows = self.run_npy(args, jtl(prstart=prstart, prstep=prstep), "dec.npy")
                self.assertEqual(names, full_names)
                self.assertGreater(len(rows), 10)
                for row in rows:
                    ref = index.get(round(row[0] * 1e16))
                    self.assertIsNotNone(ref, f"{args} {prstart} {prstep}: no full-run row at {row[0]}")
                    for x, y in zip(row, ref):
                        self.assertLessEqual(abs(x - y), 1e-12 * abs(y) + 1e-18, f"{args} {prstart} {prstep}")

    def run_npy(self, args, deck, name="full.npy"):
        self.run_josim(["-m", "1", *args, "-o", name], deck)
        return read_npy(self.dir / name)

    # ---- Monte Carlo and sweeps ----
    def test_monte_carlo_batches(self):
        nominal_names, nominal = self.waveform([], jtl())
        for args in (["-b", "4"], ["-n", "-b", "4"]):
            names, rows = self.waveform(args, jtl(spread=".spread 0"), "mc0.csv")
            for trace in ("P(B1)", "P(B3)"):
                mean, std = names.index(trace), names.index(trace + "_STD")
                low, high = names.index(trace + "_MIN"), names.index(trace + "_MAX")
                ref = nominal_names.index(trace)
                for row, nom in zip(rows, nominal):
                    self.assertLessEqual(abs(row[mean] - nom[ref]), 1e-6 * abs(nom[ref]) + 1e-12, args)
                    self.assertLessEqual(abs(row[std]), 1e-6 * abs(nom[ref]) + 1e-12, args)
                    self.assertLessEqual(row[low], row[mean] + 1e-12)
                    self.assertLessEqual(row[mean], row[high] + 1e-12)
        spread = {}
        for args in (["-b", "4"], ["-n", "-b", "4"]):
            output = self.run_josim(["-m", "1", *args, "-o", "mc.csv"], jtl(spread=".spread 0.05"))
            names, rows = read_csv(self.dir / "mc.csv")
            std = names.index("P(B3)_STD")
            self.assertGreater(max(row[std] for row in rows), 0.0, args)
            spread[" ".join(args)] = rows[-1][names.index("P(B3)")]
        self.assertLess(abs(spread["-b 4"] - spread["-n -b 4"]), 1e-6 * abs(spread["-b 4"]) + 1e-9)

    def test_parameter_sweeps(self):
        for args in (["-S", "@BIAS:0.9:1.1:0.1"], ["-n", "-S", "@BIAS:0.9:1.1:0.1"],
                     ["-S", "@BIAS:MARGIN:0.3:5"], ["-n", "-S", "BIAS:160u:190u:10u"]):
            names, rows = self.waveform(args, jtl(), "sw.csv")
            self.assertIn("P(B3)_STD", names, args)
            self.assertGreater(len(rows), 100)
        self.run_josim(["-m", "1", "-n", "-S", "@BIAS:0.9:1.1:0.1", "-e", "swe.csv"], jtl())
        with open(self.dir / "swe.csv", newline="") as stream:
            self.assertGreater(len(list(csv.reader(stream))), 3)

    def test_margins(self):
        for args in (["-M", "@BIAS:0.4:9", "-F", "m.csv"], ["-n", "-M", "@DEVICES:0.4:9", "--margin-file=m2.csv"],
                     ["-M", "@BIAS,@JJ:0.5:7", "-T", "5p"]):
            output = self.run_josim(args, jtl())
            self.assertIn("Margin", output, args)
        for name in ("m.csv", "m2.csv"):
            with open(self.dir / name, newline="") as stream:
                self.assertGreater(len(list(csv.reader(stream))), 1, name)

    def test_rich_deck_on_every_path(self):
        """Every device kind on both sides of a line: the partitioned windowed loop, the nodal partitions and the
        serial Monte Carlo loop (no line) against the plain MNA loop, in both analysis types."""
        for analysis in ("1", "0"):
            with self.subTest(analysis=analysis):
                reference = self.events(["-a", analysis], RICH)
                self.assertTrue(any(reference.values()))
                # -g without CUDA falls back to the CPU partitioned windowed loop (also without OpenMP).
                for args in (["-p"], ["-n"], ["-p", "-n"], ["-c", "2"], ["-g"]):
                    self.assert_events_match(reference, self.events([*args, "-a", analysis], RICH), 1e-13,
                                             f"rich -a {analysis} {' '.join(args)}")
                names, rows = self.waveform(["-a", analysis], RICH, "rich.csv")
                for args in (["-p"], ["-p", "-n"], ["-g"]):
                    other_names, other = self.waveform([*args, "-a", analysis], RICH, "rich2.csv")
                    self.assertEqual(other_names, names)
                    for c in range(1, len(names)):
                        scale = max(abs(row[c]) for row in rows) or 1.0
                        worst = max(abs(a[c] - b[c]) for a, b in zip(rows, other))
                        self.assertLess(worst, 1e-6 * scale + 1e-15, f"{names[c]} -a {analysis} {args}")
        plain = RICH.replace("T1 2 0 3 0 TD=3p Z0=2", "LT 2 3 1p")
        for args in (["-b", "3"], ["-n", "-b", "3"], ["-c", "1", "-b", "3"]):
            nominal_names, nominal = self.waveform(args[:-2], plain, "plain.csv")
            names, rows = self.waveform(args, plain, "plainmc.csv")
            for trace in ("P(B1)", "P(B2)"):
                mean, ref = names.index(trace), nominal_names.index(trace)
                for row, nom in zip(rows, nominal):
                    self.assertLessEqual(abs(row[mean] - nom[ref]), 1e-6 * abs(nom[ref]) + 1e-9, args)
            names, rows = self.waveform(args, plain.replace(".spread 0\n", ".spread 0.03\n"), "plainmc.csv")
            self.assertGreater(max(row[names.index("P(B1)_STD")] for row in rows), 0.0, args)

    def test_long_chain_nodal_partitions(self):
        """A 40-stage JTL split by a line: large enough for the nodal solver's partitioned windowed loop."""
        lines = [".model jmod jj(rtype=1, vg=2.8mV, cap=0.07pF, r0=160ohm, rn=16ohm, icrit=0.25mA)",
                 "IIN 0 n0 pwl(0 0 20p 0 22p 500u 24p 0 60p 0 62p 500u 64p 0)"]
        node = "n0"
        for k in range(40):
            if k == 20:
                lines.append(f"T1 {node} 0 m0 0 TD=4p Z0=2")
                node = "m0"
            lines += [f"B{k} {node} 0 jmod", f"RS{k} {node} 0 3", f"IB{k} 0 {node} pwl(0 0 10p 175u)",
                      f"L{k} {node} s{k} 2p"]
            node = f"s{k}"
        lines += [f"RL {node} 0 5", ".tran 0.25p 160p 0 0.25p", ".print p(B0) p(B19) p(B20) p(B39)", ".end"]
        deck = "* long chain\n" + "\n".join(lines) + "\n"
        reference = self.events([], deck)
        self.assertEqual(len(reference.get("P(B39)", [])), 2, reference)
        for args in (["-n"], ["-n", "-a", "0"], ["-p", "-n"], ["-g"]):
            tolerance = 0.2e-12 if "0" in args else 1e-13
            self.assert_events_match(reference, self.events(args, deck), tolerance, " ".join(args))
        output = self.run_josim(["-V", "1", "-n", "-o", "chain.csv"], deck)
        names, rows = read_csv(self.dir / "chain.csv")
        self.assertEqual(len(rows), 640, output[-1500:])

    def test_voltage_analysis_phase_output(self):
        deck = jtl(extra=".print phase B2\n.print nodep 5 0\n.print nodev 3 0")
        phase_names, phase_rows = self.waveform(["-a", "1"], deck, "a1.csv")
        for args in (["-a", "0"], ["-n", "-a", "0"]):
            names, rows = self.waveform(args, deck, "a0.csv")
            self.assertEqual(names, phase_names)
            for trace in ("P(B1)", "P(B2)", "P(5,0)"):
                c, d = names.index(trace), phase_names.index(trace)
                worst = max(abs(a[c] - b[d]) for a, b in zip(rows, phase_rows))
                self.assertLess(worst, 0.05, f"{trace} {args}")

    def test_sweep_margins_and_assertions_in_batches(self):
        for args, name in ((["-S", "@BIAS:MARGIN:0.4:9", "-T", "3p"], "swm.csv"),
                           (["-n", "-S", "@BIAS:0.8:1.2:0.1", "-S", "@JJ:0.9:1.1:0.1"], "sw2.csv")):
            self.run_josim([*args, "-e", name], jtl())
            with open(self.dir / name, newline="") as stream:
                self.assertGreater(len(list(csv.reader(stream))), 6, args)
        good = ["-A", "B3:COUNT=2", "-A", "B3:TIMES=36.9p,76.9p TOL=3p", "-A", "B3:AFTER=B1 DELAY=4.8p TOL=2p",
                "-A", "B3:TD=20p PER=40p BITS=11"]
        for args in (["-b", "3"], ["-n", "-b", "3"], []):
            output = self.run_josim([*args, *good], jtl(spread=".spread 0.01"))
            self.assertIn("Assertions: PASS", output, args)
        output = self.run_josim(["-n", "-b", "3", "-A", "B3:COUNT=5", "-A", "B3:TIMES=10p TOL=1p",
                                 "-A", "B3:TD=20p PER=40p BITS=111", "-A", "B3:AFTER=B1 DELAY=30p TOL=1p"],
                                jtl(spread=".spread 0.01"), success=False)
        self.assertIn("FAIL", output)

    def test_margin_targets(self):
        for args in (["-M", "@BIAS,@JJ_IC,@IND,@RES:0.5:7"], ["-n", "-M", "B1,L2,RS3:0.4:7"],
                     ["-n", "-M", "@DEFAULT", "-T", "2p"]):
            output = self.run_josim(args, jtl())
            self.assertIn("Margin", output, args)

    def test_netlist_controls(self):
        """The analyses and options written as netlist controls instead of command-line switches."""
        decks = (
            (".margin @BIAS RANGE=0.4 N=7 TOL=3p FILE=dm.csv\n.margin @JJ MIN=0.6 MAX=1.4 STEP=0.2", "Margin", "dm.csv"),
            (".margin @BIAS 0.3 5 TRACES=P(B3)\n.margin @IND -0.3 0.3", "Margin", None),
            (".sweep @BIAS 0.9 1.1 0.1", None, None),
            (".sweep @BIAS MARGIN 0.3 5", None, None),
            (".sweep @BIAS MARGIN -0.2 0.2 5", None, None),
            (".sweep BIAS LIST 160u 175u 190u", None, None),
            (".assert B3 count=2\n.assert B3 td=20p per=40p bits=11\n.assert B3 after B1 delay=4.8p tol=2p\n"
             ".assert B3 times=36.9p,76.9p tol=3p\n.assert B3 td=20p per=40p bits=0xC", "Assertions: PASS", None),
            (".meas tran T_PROP TRIG P(B1) SFQ=1 TARG P(B3) SFQ=1\n.meas tran T_POL TRIG P(B1) SFQ=1 POL=1 "
             "TARG P(B3) SFQ=2 POL=1\n.meas tran N_B3 COUNT B3 VAL=2\n.meas tran P2 PARAM='T_PROP*1e12' MIN=1 MAX=20",
             "Measurement Report", None),
            (".jitter J_PROP B3 REF=B1 MAX_RMS=1p STRICT\n.jitter J_BITS B3 TD=20p PER=40p BITS=0x3", "Jitter", None),
            (".meas tran T_PROP TRIG P(B1) SFQ=1 TARG P(B3) SFQ=1\n.sens @BIAS DELTA=5% METRIC=T_PROP N=5\n"
             ".sens @JJ_IC DELTA=0.05 METRIC=T_PROP CENTER MIN=0.6 MAX=1.4 STEP=0.2", "Sensitivity", None),
            (".option seed=12345 nodal txinterp warnstep\n.option corrector=2 correctortol=1e-4\n.option kcutoff=1e-3",
             None, None),
            (".option eventsonly eventmetrics=all checkloops\n.option nolumpptl", None, None),
            (".option events lumpptl\n.option nosens nocheckloops noevents", None, None),
        )
        for controls, expected, produced in decks:
            with self.subTest(controls=controls.split("\n")[0]):
                output = self.run_josim([], jtl(extra=controls))
                if expected:
                    self.assertIn(expected, output, output[-2500:])
                if produced:
                    self.assertGreater((self.dir / produced).stat().st_size, 0)
        self.run_josim(["-m", "1", "-e", "golden.csv"], jtl())
        output = self.run_josim([], jtl(extra=".verify_events golden.csv TOL=1p"))
        self.assertIn("Event Verification: PASS", output)

    def test_sensitivity(self):
        output = self.run_josim(["--sens=@BIAS,@JJ", "--sens-file=s.json"], jtl())
        self.assertIn("Sensitivity", output)
        self.assertTrue((self.dir / "s.json").stat().st_size > 0)
        output = self.run_josim(["-n", "--sens=@DEVICES CENTER", "--sens-file=s.csv"], jtl())
        self.assertIn("Sensitivity", output)

    # ---- state files and initial conditions ----
    def test_saved_state_continues_the_run(self):
        pulse = "pwl(0 0 20p 0 22p 300u 24p 0 80p 0 82p 300u 84p 0 140p 0 142p 300u 144p 0)"
        continuous = jtl(tstop="120p", bias="dc BIAS", pulse=pulse)
        for args in ([], ["-n"]):
            names, whole = self.waveform(args, continuous, "whole.csv")
            self.waveform(args, jtl(tstop="60p", bias="dc BIAS", pulse=pulse, extra=".savestate state.bin"), "a.csv")
            self.assertGreater((self.dir / "state.bin").stat().st_size, 0)
            _, resumed = self.waveform(args, jtl(tstop="60p", bias="dc BIAS", pulse=pulse,
                                                 extra=".loadstate state.bin"), "b.csv")
            later = {round(row[0] * 1e16): row for row in whole}
            trace = names.index("P(B3)")
            for row in resumed[4:]:
                ref = later.get(round((row[0] + 60e-12) * 1e16))
                if ref is None:
                    continue
                self.assertLess(abs(row[trace] - ref[trace]), 2e-3, f"{args} at {row[0]}")

    def test_initial_conditions(self):
        for args in (["--ic=P(5)=6.2832"], ["-n", "--ic=P(5)=6.2832 V(1)=0"]):
            names, rows = self.waveform(args, jtl(), "ic.csv")
            self.assertGreater(rows[0][names.index("P(B3)")], 3.0, args)
        output = self.run_josim(["--check-loops=loops.csv"], TESTS / "syntax" / "test_ic_flux_loops.cir")
        self.assertIn("Loop (beta_L) Audit Report (3 loops, 1 storage, 0 degenerate)", output)
        self.assertGreater((self.dir / "loops.csv").stat().st_size, 0)

    # ---- analysis decks the integration suite does not run ----
    def test_feature_decks(self):
        for name in ("test_meas.cir", "test_jitter_ber.cir", "test_tx_loss.cir", "test_ic_flux_loops.cir"):
            with self.subTest(deck=name):
                output = self.run_josim(["--meas-file=meas.csv", "--jitter-file=jit.json"],
                                        TESTS / "syntax" / name)
                self.assertNotIn("FAIL", output.replace("FAILED=0", ""), output[-2500:])
        output = self.run_josim(["-J", "B2|X3 REF=B2|X4 MAX_RMS=1p", "--jitter-file=j.csv"], DFF)
        self.assertIn("Jitter", output)
        output = self.run_josim(["--meas=T_SW TRIG P(B2|X3) SFQ=1 TARG P(B2|X4) SFQ=1"], DFF)
        self.assertIn("T_SW", output.upper())

    def test_iv_and_squid_sweeps(self):
        output = self.run_josim([], TESTS / "syntax" / "test_iv_squid.cir", timeout=900)
        self.assertNotIn("FAIL", output.replace("FAILED=0", ""), output[-2500:])

    def test_standard_input_and_sanity(self):
        output = self.run_josim(["-i", "-m", "1", "-o", "stdin.csv"], stdin=JTL.read_text())
        self.assertTrue((self.dir / "stdin.csv").stat().st_size > 0, output)
        self.run_josim(["-s"], JTL)

    # ---- diagnostics ----
    def test_bad_input_is_reported(self):
        base = jtl()
        cases = (
            (base.replace("pwl(0 0 10p BIAS)\nIB2", "pwl(0 0 10p)\nIB2"), [], "PWL"),
            (base.replace(PULSES, "sin()"), [], "SIN"),
            (base.replace(PULSES, "pwl(0 0 1p)"), [], "PWL"),
            (base.replace("B2 3 0 jmod", "B2 3 0 nomodel"), [], "nomodel"),
            (base.replace("L2 3 5 2p", "L2 3"), [], "L2"),
            (base.replace(".end", "K1 L1 L9 0.5\n.end"), [], "L9"),
            (base.replace(".tran 0.25p 120p 0 0.25p", ".tran"), [], "TRAN"),
            (base.replace("rtype=1,", "rtype=7,"), [], "RTYPE"),
            (base.replace("B3 5 0 jmod", "X1 undefined_sub 5 0"), [], "undefined_sub"),
            (base.replace(".end", ".include missing_file.cir\n.end"), [], "missing_file"),
            (base, ["-R"], "-R"),
            (base, ["-a", "5"], "nalysis"),
            (base, ["-V", "x"], "erbos"),
            (base, ["-S", "@BIAS:bad"], "bad"),
            (base, ["--verify-events=missing.csv"], "missing.csv"),
            (base.replace(".print p(B1)", ".print p(B9)"), [], "B9"),
            (base.replace(".param BIAS=175u", ".param BIAS={1/0}"), [], "1/0"),
            (base.replace(".param BIAS=175u", ".param BIAS={UNDEF*2}"), [], "BIAS"),
            (base.replace(PULSES, "pat(1u 0 2 1p 1p 10p b2x1)"), [], "B2X1"),
            (base.replace(PULSES, "exp(1u)"), [], "EXP("),
            (base.replace(PULSES, "cus(missing.txt 1p 1 0 0)"), [], "CUS"),
            (base.replace(".end", "Z1 1 0 5\n.end"), [], "Z1"),
            (base.replace(".end", "RS1 1 0 3\n.end"), [], "Duplicate label RS1"),
            (base.replace(".end", "C9 1 0 0\n.end"), [], "matrix"),
            (base.replace(".end", "E1 7 0 1 0\nR7 7 0 1\n.end"), [], "E1 7 0 1 0"),
            (base.replace(".end", ".subckt S a b\nR1 a b 1\n.ends\nX1 S 1\n.end"), [], "X1"),
            (base.replace(".end", ".model mm foo(r=1)\n.end"), [], "Unknown model type"),
            (base.replace(".tran 0.25p 120p 0 0.25p", ".tran 0.25p -120p"), [], "positive"),
            (base.replace(".print p(B1) p(B2) p(B3) i(L3)", ".print devv"), [], "DEVV"),
            (base.replace("T1 2 0 3 0 TD=3p Z0=2", "T1 2 0 TD=3p Z0=2"), [], "T1 2 0"),
            (base.replace(".param BIAS=175u", ".spread abc\n.param BIAS=175u"), [], "ABC"),
            (base.replace(".end", ".neb abc\n.end"), [], "ABC"),
            (base.replace(".end", ".temp abc\n.end"), [], "ABC"),
            (base.replace(".end", ".option corrector=abc\n.end"), [], "ABC"),
            (base.replace(".end", ".file\n.end"), [], ".file"),
        )
        for deck, args, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic, args=args):
                output = self.run_josim(["-m", "1", *args, "-o", "bad.csv"], deck, success=False)
                self.assertIn(diagnostic.lower(), output.lower(), output[-1500:])


if __name__ == "__main__":
    unittest.main()
