# Upstream JoSIM 3.1.0 merged into the fork

Branch `merge-upstream-3.1.0`: the fork (upstream v2.7 + 34 commits, 05f6494) with the NOISE grid fix (736e609)
merged with upstream `febd56f` (3.1.0 plus one unreleased commit, 79 commits after v2.7).

## What stays the fork's

The fork's solvers and speed paths are unchanged: the nodal solver (`-n`), partitioned and OpenMP paths (`-p`), GPU
paths (`-g`), Monte Carlo/`.spread`, margins (`-M`), batched sweeps (`.sweep`, `-S`), `.sens`, events-only output
(`-e`), print-grid decimation, rollback sub-steps, binary output, co-simulation (`libjosim_c`, `josim-cosim-server`,
Spectre DPI, ngspice code model), PAT/SFQ sources, nTron/hTron devices and the fork's CLI options.

## What comes from upstream

- **Noise.** NOISE sources advance on their own grid (the fork's 736e609 is the same loop) with the amplitude applied
  after interpolation; thermal noise is sampled once per step, after a junction's conductance for that step is chosen;
  PNOISE/FNOISE/ICNOISE; seeded runs reset the Gaussian cache.
- **Junctions.** One Josephson scale (SIGMA) in the matrix, the history equations and the fork's timestep rescales
  (upstream #91); histories used during startup and at t = 0; gate-controlled junctions (`GC=`), `BETA_C`, model
  domain validation; transparent and zero-temperature limits.
- **Inductors.** Voltage-mode mutual inductors read the same preceding currents (the fork's MNA loop too); nonlinear
  inductors (`ALPHA=`).
- **Capacitor `IC=`** (needs `.tran ... DST`), phase-source startup histories, TX launch-voltage correction.
- **Netlist.** One-pass expansion with parameters and models bound per instance, header defaults, `PARAMS:`, pin
  aliases in traces (`V(IN.X1)`), recursion errors. The expression evaluator is upstream's (strict, unary signs,
  `^` precedence, suffix checks), extended with the fork's functions (1-3 arguments, variadic MIN/MAX).
- **Output.** Print points on an exact grid (`prstart + k prstep` with a few-ulp tie tolerance, upstream e6e0cbf) in
  every writer and in the decimation schedule; `V(A,B)`/`I(A,B)` of two devices as a difference; current sources
  report the samples they applied; write and flush failures and non-finite results are errors.
- **CLI and input.** Unknown switches, empty arguments and values on flags are errors; `--verbose` no longer maps to
  `--version`; output files may not overwrite inputs; cyclic includes are errors.
- **Model IV.** The classic form `.iv <model> <Imax> <file> [STEP= TSTEP= TSTOP= TSTART= GTJJ=]` has upstream's
  semantics (fresh state per point, sweep ends exactly at +-Imax); points still run in parallel. The fork's keyword
  forms are unchanged but reject unknown keys.
- **Build.** C++17/C11, pinned SuiteSparse archive, CMake package export (with the OpenMP dependency), upstream's
  regression suites.

## The reference MNA path

Decks that use what only upstream models -- `GC=` or `ICNOISE=` junctions, `ALPHA=` inductors, capacitor `IC=`,
PNOISE/FNOISE/ICNOISE sources, NOISE-driven voltage or phase sources, or temperature-dependent junction models
(`T=`, `TC=`, `D=`) -- run on `trans_sim_reference` (`src/SimulationReference.cpp`): upstream's transient loop and step
helpers on the device state in the matrix components, with the fork's transmission-line and nanowire handlers. These
decks do not use the nodal, partitioned, GPU or Monte Carlo paths.

## NEB

| | Draw interval of automatic thermal noise | Bandwidth of the interpolated process |
|---|---|---|
| fork <= 05f6494 (+736e609) | 1/NEB | NEB/3 |
| this merge (upstream 3.1) | max(step, 1/(3 NEB)) | NEB (until the step floor) |

The low-frequency density is 4kT/R in both, so a thermal bath at frequencies well below NEB is unchanged; the noise
now extends three times higher at the same `.neb`. `.neb` set to a third of the old value reproduces the old spectrum.
An explicit `NOISE(VA TD)` without TSTEP defaults to 1/(3 NEB) when `.neb` is given (before: the simulation step).
Seeded noise waveforms differ from 05f6494's draw for draw.

## Differences from 05f6494 on noise-free decks

- **Print grid.** After a step halving (or whenever accumulated print times hit a rounding tie), 05f6494 printed some
  points one internal step late and could drop the last point; this build prints the exact grid. Waveform rows can
  move by one internal step, and events-only results (detected on the kept print-grid samples) move by femtoseconds.
- **Josephson scale.** SIGMA instead of HBAR/(2 EV) in the matrix: last-bit differences (about 3e-10 relative).
- **Startup.** Only decks with non-zero sources before t = 0 run startup steps; for them, junction and phase-source
  histories now carry through startup (upstream 3.0).
- **Voltage-mode mutual inductors**: coupled branches read the same preceding currents.

## Regression on the flow's bench decks (exxact, one thread, 05f6494 `-n` vs 7bbcde7 `-n`, 2026-11-19)

| Deck | Events | Worst event shift | Wall s, waveforms (05f6494 / merge) | Wall s, events-only | Peak MB, waveforms |
|---|---:|---:|---|---|---|
| dff | 9 | 0.000 fs | 0.32 / 0.19 | 0.31 / 0.15 | 13 / 13 |
| xadd4 | 56 | 0.000 fs | 11.77 / 11.63 | 9.78 / 9.92 | 146 / 149 |
| r10f1n | 4 | 0.000 fs | 27.45 / 28.20 | 28.66 / 27.92 | 253 / 257 |
| r10forig | 4 | 0.000 fs | 3.51 / 3.50 | 3.57 / 3.49 | 243 / 247 |
| r5cal | 211 | 0.001 fs | 82.23 / 84.97 | 83.39 / 83.70 | 209 / 214 |
| coupler | 12,784 | 0.061 fs | 35.86 / 35.96 | 11.23 / 11.05 | 1404 / 1407 |
| adder | 11,920 | 0.000 fs | 82.07 / 81.62 | 22.95 / 22.94 | 1719 / 1722 |
| r12p15 (Q5 macro) | 5,912 | 0.002 fs | 1373.8 / 1385.0 | 1390.3 / 1398.6 | 771 / 775 |

Every deck keeps every switching event; the events-only extractor lists exactly the events in the waveforms. Each
output has one more row (the corrected grid keeps the last print point). Sample differences are in the last digits
(Josephson scale). Runs were paired side by side; differences of 1-3 % are within their spread.

## Upstream tests adapted to fork behavior

- `regression_cli.py`: `-p` is valid (the fork's OpenMP solvers); the help lists more long options; a step too coarse
  for a junction is refined locally (rollback sub-steps) instead of restarting with a halved global step, so the
  reported grid is the requested one.
- `cmake/test_rejected_options.cmake`: `USING_OPENMP=ON` is supported.
- `test/CMakeLists.txt` (fork's pyjosim test): 1200 samples, the corrected print grid (05f6494 printed 1199).
