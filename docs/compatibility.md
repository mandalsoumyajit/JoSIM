# Updating to 3.1

This fork is built from source ([build guide](index.md)); upstream's packages are on
[GitHub Releases](https://github.com/JoeyDelp/JoSIM/releases).

## Changes from earlier fork builds (v2.8, up to 05f6494 and 736e609)

The fork now contains upstream 3.1.0 ([merge notes](upstream-3.1-merge.md)). The fork's solvers, options and output
formats are unchanged; these changes can affect existing decks and results:

- **Print grid.** Points sit on the exact grid `prstart + k·prstep`. After a step halving some rows move by one
  internal step, every output keeps its final point (one more row), and events-only times move by femtoseconds.
- **Josephson constant.** One constant (SIGMA) throughout: differences in the last digits (about 3e-10 relative).
- **NEB is the noise bandwidth.** Thermal noise draws every max(step, 1/(3 NEB)) instead of every 1/NEB, so at the
  same `.neb` it reaches three times higher in frequency; the low-frequency bath is unchanged. `.neb` at a third of
  the old value gives the old spectrum. Seeded noise does not repeat draw for draw.
- **Noise after a rollback.** When a step is too coarse for a junction, the fork rolls back to a checkpoint. Noise
  generators now restart from the checkpoint; before, they extrapolated backwards over the re-simulated interval.
- **Capacitor `IC=`** is upstream's initial voltage and needs `.tran ... DST`; `V0=` keeps the fork's old meaning.
- **Stricter input.** Unknown command-line switches, unknown `.iv` keys, malformed expressions and invalid trace
  requests are errors. `NODEV a b c` prints one trace per node.
- **Fixes.** `-p` with coupled inductors in voltage analysis, and `-b` with the corrector, thermal noise or a state
  file on a partitioned deck (it ran one trial).

## Changes from 3.0

JoSIM 3.1 adds [capacitor initial voltages](model_limits.md#capacitor-voltage),
[BETA_C for RTYPE=0 junctions](model_limits.md#beta-c) and
[quadratic nonlinear inductors](model_limits.md#nonlinear-inductance).
Capacitor `IC=` requires `.TRAN ... DST`. The 3.1.0 binaries reject phase and
controlled sources in IC runs; source builds after 3.1.0 support these sources
with consistent initial constraints. `ALPHA=` specifies differential inductance,
so its cubic self-flux term includes a factor of one third.

Voltage-mode mutual inductors now use the same preceding current samples for
all coupled branches. This correction can change results in existing coupled
circuits. Recheck timestep convergence when comparing them with 3.0.

## Changes from releases before 3.0

Several fixes change simulation results, particularly phase accumulation,
startup, subcircuit parameters, model IV curves and noise. Recheck timestep,
settling time and output sampling when comparing older results with 3.0.
A matching seed does not reproduce the same noise waveform across versions.
The [noise guide](noise_sources.md) explains the corrected bandwidth and
junction thermal-noise calculations.

Named subcircuit parameters and hierarchical pin outputs now work as described
in the [syntax guide](syntax.md#subcircuits). Invalid trace requests, malformed
waveforms and invalid model values produce errors. Output files cannot overwrite
the main netlist or one of its includes.

KLU is the supported solver; upstream removed SuperLU (`SLU=ON`, `-x`). Upstream also removed its OpenMP option,
but this fork keeps its own OpenMP solvers (`USING_OPENMP=ON`, `-p`, `-p -n`) and CUDA solvers (`USING_CUDA=ON`,
`-g`); see [usage](usage_and_benchmarks.md) and [GPU architecture](gpu_architecture.md).

The old `ivcurve.py` and `noise_insert.py` scripts are retired. Use the
[circuit-sweep helper](sweeps.md) and native `.TEMP`/`.NEB` controls instead.

## C++ applications

Rebuild applications, including those built for 3.0, using the headers and
libraries from the same 3.1 installation. Capacitor initialization and nonlinear
inductors change public type layouts. JoSIM exposes C++ types,
so the compiler and C++ runtime must also be compatible.

The CMake target is `josim::josim`:

```cmake
find_package(josim 3.1 CONFIG REQUIRED)
target_link_libraries(my_simulator PRIVATE josim::josim)
```

Shared libraries use ABI version 4, independently of the release version.
Unix applications linked to ABI version 3 must be rebuilt. See [Embedding JoSIM](API.md) for
error handling, result storage and the restrictions on concurrent runs.
