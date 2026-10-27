# JoSIM
### Superconductor Circuit Simulator

---

## Full Documentation

Please see full documentation here:
[JoSIM Documentation](https://joeydelp.github.io/JoSIM)

## Overview

JoSIM was developed under IARPA contract SuperTools(via the U.S. Army Research Office grant W911NF-17-1-0120).

JoSIM is a SPICE syntax circuit simulator specifically created to handle superconducting elements such as the Josephson junction. It reads in a standard SPICE deck, creates an A matrix and solves the linear algebra problem *Ax=b*. The linear algebra package KLU by [Tim Davis](http://faculty.cse.tamu.edu/davis/suitesparse.html) is used to solve the system of equations.

A two stage distribution platform is implemented in JoSIM with the first being a command line interface (CLI) binary that is always built as standard during compilation. This is accompanied by a C++ library (*libjosim*).

JoSIM allows output of results in standard text formats (`.csv`, `.dat`, ASCII SPICE `.raw`) as well as high-speed binary formats (`.bin` native column-major IEEE-754 float64, `.npy` NumPy structured array, and `.rawb` SPICE3 binary raw). JoSIM has inherent support for `.PARAM` commands that allow components to have variable values as well as expression parsing. JoSIM implements the RCSJ model of the Josephson junction (with optional implicit predictor-corrector iteration), lossless transmission lines (with optional fractional-step delay interpolation), and state checkpointing (`.savestate` / `.loadstate`) for fast bias conditioning.

JoSIM takes a *.cir* file as input and produces a *.csv*, *.dat*, *.raw*, *.bin*, *.npy*, or *.rawb* file as output.

Prebuilt binaries can be found here:

https://github.com/JoeyDelp/JoSIM/releases

Referencing:

[JoSIM — Superconductor SPICE Simulator](https://ieeexplore.ieee.org/document/8633946/)

If you find this tool useful, you can help keep it alive by \
[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/joeydelp) \
or via \
[![](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=%23fe8e86)](https://github.com/sponsors/JoeyDelp)

---

## Performance, Binary I/O & Enhanced Features

This enhanced fork of JoSIM delivers significant single-core, multi-threaded (`-p`), numerical integration, and binary I/O improvements (see [`ENHANCEMENT_PLAN.md`](ENHANCEMENT_PLAN.md) for architectural details):

### 1. Speed & OpenMP Parallelism (`-p` / `--parallel`)
- **2.84x faster on the 23-test CTest suite** (`20.20 s -> 7.11 s` in WSL with 100% bit-identical outputs).
- **2.2x–3.2x faster end-to-end simulation + output** on large extracted superconducting IC netlists when combining the solver optimizations with binary output (`.bin` / `.npy`).
- **$O(1)$ hash-map lookups** (`deviceMap`, `csMap`) for mutual inductances (`K`) and `.print`/`.save` traces, eliminating $O(N^2)$ setup bottlenecks on extracted layouts with thousands of mutual couplings.
- **In-place CSR updates & `klu_l_refactor`** whenever `RTYPE=1` Josephson junctions cross their gap voltage threshold ($V_g \pm \frac{1}{2}\Delta V$), avoiding per-transition matrix rebuilds and heap allocations.
- **Block-Partitioned KLU Solver across PTLs**: Automatically detects connected components of the MNA matrix separated by lossless transmission lines (`T` elements), creating independent `klu_l_symbolic`/`klu_l_numeric` sub-matrices (`KluPartition`). Only blocks containing a transitioning junction are refactored (`klu_l_refactor`), and with `-p` (`--parallel`), blocks are refactored and solved in parallel across OpenMP threads.
- **Self-Contained Transmission-Line Ring Buffers & Mid-Flight Checkpointing**: Each transmission line maintains a compact circular ring buffer (`hist_`) of length $k + 4$ instead of forcing all 6 of its port/current variables into `results.xVector` for the entire simulation length ($N_{\text{steps}}$), reducing transmission-line memory by **100x–1,000x** and supporting exact `.savestate`/`.loadstate` resumption even while SFQ pulses are mid-flight inside PTLs.
- **In-Loop Print Windowing & Decimation (`PRSTART` / `PRSTEP`)**: When `.tran` specifies `PRSTART > 0` or `PRSTEP > TSTEP` (without FIR windowing), `Simulation::trans_sim` stores only the required output steps (plus 2-step BDF2 lookback points when phase-to-voltage differentiation is requested), slashing RAM usage on long transient runs.
- **Homogeneous device arrays & zero-allocation hot loop**, cached `PWL`/`PWS` time segments, pre-computed `CUS` cubic splines, fast pre-reserved subcircuit flattening, and functional OpenMP multi-threading (`-p`) across device evaluation, block KLU solves, and output formatting.

### 2. High-Speed Binary Output Formats (`.bin`, `.npy`, `.rawb`)
Specify the output format via the file extension in `-o <file>` or `.file <file>`:
- **`.bin` (`JOSIMBIN`)**: Native column-major IEEE-754 `float64` binary format. Performs a zero-copy contiguous `fwrite` per trace directly from `Trace::data_`, making pure file writing **~37x faster than CSV** and reducing file size by ~38% while preserving full 64-bit precision.
- **`.npy`**: Standard NumPy v1.0/v2.0 structured array with named `<f8` columns (`time`, `"V(1)"`, etc.). Loadable directly in Python via `np.load("out.npy")` up to **23x faster than parsing CSV**.
- **`.rawb`**: Standard SPICE3 binary `.raw` format (`Binary:\n` preamble followed by interleaved `float64` rows).

Use [`scripts/josim_io.py`](scripts/josim_io.py) in Python to load `.bin`, `.npy`, or `.csv` files into a dict of NumPy arrays or a Pandas `DataFrame`:
```python
from scripts.josim_io import load_josim, load_josim_bin, load_josim_npy

data = load_josim("output.bin")          # Returns dict[str, np.ndarray]
df = load_josim("output.bin", as_dataframe=True)
arr = load_josim_npy("output.npy")       # Zero-copy memory-mapped structured array
```

### 3. New CLI Switches & Netlist Controls
- **Fractional Transmission-Line Delay Interpolation (`.option txinterp` / `INTERP=1`)**:
  Linearly interpolates delayed transmission-line port states when $T_d / T_{\text{step}}$ is non-integer, eliminating delay-rounding waveform drift across timestep sweeps:
  ```spice
  .option txinterp
  T01 1 0 2 0 td=5.07p z0=50 interp=1
  ```
- **State Checkpointing (`.savestate` / `.loadstate`)**:
  Save the complete MNA solution vector and all multi-step device & transmission-line histories after bias conditioning, and reload instantly in subsequent runs:
  ```spice
  .savestate bias_6ns.state 6000p
  * In subsequent runs:
  .loadstate bias_6ns.state
  ```
  (`.savebias` and `.loadbias` are also supported as aliases.)
- **Implicit Predictor-Corrector JJ Iteration (`-c [N]` / `.option corrector=N`)**:
  Performs up to `N` damped Picard/BDF2 corrector iterations per timestep on active Josephson junctions (with optional `.option correctortol=<tol>`, default `1e-3` rad).
- **Fixed-Step Enforcement & Step-Halving Diagnostics (`-f` / `.option fixedstep` / `.option warnstep`)**:
  Disable automatic step-halving restarts (`-f` / `--fixed-step` or `.option fixedstep`) or print detailed diagnostics naming the junction that triggered step reduction (`.option warnstep` or `-V 1`).
- **Mutual Inductance Coupling Cutoff (`.option kcutoff=<val>`)**:
  Automatically prunes parasitic mutual inductance entries (`K` lines) with $|k| < \text{kcutoff}$ during matrix construction to prevent dense LU fill-in on 3D field-extracted (e.g. InductEx) netlists:
  ```spice
  .option kcutoff=1e-3
  ```

### 4. WSL Build, Benchmark & Verification Scripts
- [`scripts/build_wsl.sh`](scripts/build_wsl.sh): Configures and builds JoSIM in WSL using Ninja (`-DCMAKE_BUILD_TYPE=Release -DPARALLEL=ON`) and runs the 23-test CTest suite.
- [`scripts/benchmark_wsl.sh`](scripts/benchmark_wsl.sh): Benchmarks upstream vs. optimized JoSIM and verifies 100% bit-identical CSV outputs.
- [`scripts/test_phase3_wsl.sh`](scripts/test_phase3_wsl.sh): Tests and benchmarks `.bin`/`.npy`/`.rawb` binary outputs, fractional TX delay interpolation, `.savestate`/`.loadstate` checkpointing, `-f` fixed-step control, and `-c` JJ predictor-corrector iteration.
- [`scripts/test_phase4_wsl.sh`](scripts/test_phase4_wsl.sh): Tests transmission-line circular ring buffers, mid-flight PTL `.savestate`/`.loadstate` checkpointing, in-loop `PRSTART`/`PRSTEP` decimation, `.option kcutoff`, and OpenMP block-partitioned KLU solving across PTL boundaries.

---

## Changelog
### v2.8
- Added high-speed binary output formats: `.bin` (column-major `JOSIMBIN` float64), `.npy` (NumPy structured array), and `.rawb` (SPICE3 binary raw), along with Python loader [`scripts/josim_io.py`](scripts/josim_io.py).
- Implemented $O(1)$ hash-map lookups for mutual inductance (`K`) and `.print`/`.save` trace resolution.
- Implemented in-place CSR value updates and `klu_l_refactor` on `RTYPE=1` Josephson junction state transitions.
- Implemented block-partitioned KLU solver (`KluPartition`) across lossless transmission lines (`T` elements) with selective per-block `klu_l_refactor` and OpenMP parallel block solves (`-p`).
- Replaced full-history `results.xVector` forcing for transmission lines with self-contained circular ring buffers (`hist_`) of length $k + 4$, and added v2 `.savestate`/`.loadstate` serialization for mid-flight PTL pulses.
- Implemented in-loop `PRSTART`/`PRSTEP` print windowing and decimation in `Simulation::trans_sim` to avoid storing unprinted transient steps in RAM.
- Added `.option kcutoff=<val>` (`kmin=<val>`) to prune weak parasitic mutual inductances during matrix setup.
- Partitioned devices into contiguous homogeneous arrays (`Simulation::setup`) and eliminated per-timestep heap allocations in `Simulation::trans_sim` and `Simulation::setup_b`.
- Cached `PWL`/`PWS` active segment indices for $O(1)$ amortized source evaluation and pre-computed `CUS` cubic splines once during setup.
- Replaced per-line/per-token `std::regex` construction in netlist parsing with linear whitespace scanning, optimized hierarchical subcircuit flattening, and fixed `ProgressBar` busy-wait CPU spin.
- Fixed `Resistor::update_timestep` scaling factor in Phase mode (`factor * factor`).
- Implemented OpenMP parallelization (`-p` / `--parallel`) across device RHS updates, block-partitioned KLU solves, and output formatting (`Output::write_output`).
- Added fractional delay interpolation for transmission lines via `.option txinterp` (`.option fractx`) or per-element `interp=1`.
- Added binary simulation state checkpointing via `.savestate <file> [time]` (`.savebias`) and `.loadstate <file>` (`.loadbias`).
- Added implicit predictor-corrector JJ iteration via `-c [N]` (`--corrector=N`), `.option corrector=<N>`, and `.option correctortol=<tol>`.
- Added explicit step-halving control and diagnostics via `-f` (`--fixed-step`), `.option fixedstep`, and `.option warnstep`.
### v2.7
- Added ability to specify seed for RNG through `.option seed=` command.
- Current run seed can be shown using verbose command.
- Added piece-wise sinusoidal (PWS) syntax to syntax guide documentation.
- Expanded sinusoid parameters to include phase offset and number of cycles. Now allows non-continuous sinusoid.
- Collection of bug fixes, code & code cleanup.
- Cleaned up warnings during compilation.

### v2.6.10
- Fixed a bug resulting in sub optimal performance in the custom function.

### v2.6.9
- Fixed a bug resulting in bad optional access when shorting transmission line components.

### v2.6.8
- Add a fix to warn of subcircuit redefenitions. Redefintions now overwrite the previous definition.
- Add a fix when parameters are used in subcircuits but defined within the global scope.

### v2.6.7
- Added ability to parameterize time points specified for `.tran` and `pwl` commands.

### v2.6.6
- Fixed a bug in noise source generation when noise step is less than or equal to the simulation step.
- Reduced aggressiveness of timestep scale down when close to transmission line step delay size. Now only scales down if TX delay is less than 1 step (previously 4)
- Added output dimensions option to josim-plot script when writing directly to PDF, SVG, EPS, etc.

### v2.6.5
- Updated the CPR specification in the JJ model to allow multiple simulataneous harmonics.
- Added a Notepad++ UDL syntax highlighting script to the scripts folder.

### v2.6.4
- Added the ability to specify CPR in the JJ model to enable non-sinusoidal CPRs needed for pi-junctions.

### v2.6.3
- Fixed a bug where Windows and CentOS parameters do not coincide. Issue was due to expression substitution.

### v2.6.2
- Fixed sp_generator.py having invalid tab spacing
- Fixed Area= and Ic= JJ parameters not producing the same result for the same critical current

### v2.6.1
- Fixed an issue with custom waveform where it wasn't reading from the file properly
- Added spline method for cubic interpolation of custom waveform
- Moved entire codebase to 64bit integers to allow larger/longer simulations
- Fixed parameters
- Fixed transmission line integer step delay rounding issue that caused an error build up

### v2.6
- Full Johnson-Nyquist temperature noise for Resistors and Resistive branch of RCSJ model
- Phi-junction model with verified Pi junction results
- Various bug fixes and code clean up

### v2.5.8
- Updated KLU to a 64-bit version capable of handling non-zero count larger than the maximum value of a 32-bit integer.
- Included SuperLU 64-bit that can be activated using the *-x 1* switch (default 0 for KLU).
- Bug fixes to pulse command.

### v2.5.7
- Fixed an issue with comparison of double values for small numbers between various compilers.

### v2.5.6
- Fixed a initial values when converting between phase and voltage when simulation output does not start at 0.

### v2.5.5
- Fixed a bug in output conversion when plotting voltage in a phase simulation and vice versa.

### v2.5.4
- Fixed comparison function for GCC interpretation of double values

### v2.5.3
- Fixed an issue with the pulse command producing incorrect pulses
- Fixed an issue where timesteps smaller than 10fs were causing out-of-range issues
- Various other bug fixes

### v2.5.2
- Fixed a major issue with relative paths not being correctly evaluated causing files to not be found even though the correct path was specified.

### v2.5.1 - 18/05/2021
- Added .spread command. See documentation for more details.
- Added .file command. See documentation for more detauls.
- Added .iv command. See documentation for more details.
- Fixed a bug where .neb value was not being read properly
- Fixed a bug where pulse source type was outputting incorrect number of pulses.
- Fixed a bug where custom waveform was not allowing cubic interpolation method.
- Code cleanup.

### v2.5 - 05/03/2021
- Phase mode is now the standard analysis mode. Produces the exact same results as voltage.
- Trapezoidal method has been replaced by superior 2nd order Gear method.
- JoSIM now internally determines the convention. (-c option removed)
- New progress indicators added. Spawns a seperate non blocking thread.
- New verbose modes. 1: Circuit statistics. 2: 1 + Parameters and evaluated values. 3: 2 + Complete expanded netlist.
- New DC source type. I.e. V01 1 0 DC 5V
- JJ has new IC= parameters that supersedes AREA= but ultimately performs similar model parameter scaling.
- Fixed NOISE source type to have syntax NOISE(AMP TD TSTEP)
- New '.temp' and '.neb' commands for noise thermal noise analysis
- Transient simulation command now allows PRSTART and PRSTEP commands. .tran TSTEP TSTOP PRSTART PSTEP
- Various memory improvements and overall speed-up
- Reduced output file size.
- Rewrote documentation: [JoSIM Documentation](https://joeydelp.github.io/JoSIM)
- Various bug fixes

### v2.4.1 - 09/03/2020
- Fixed an issue where "GND" was not recognized as ground, creating a new node causing matrix singularity.
- Fixed an issue with multiport devices (TX, CCCS, CCVS, VCCS, VCVS) not adding the 3rd and 4th node to the node map.
- Fixed an issue with multiport devices not expanding properly if part of subcircuits.
- Various other bug fixes

### v2.4 - 07/02/2020
- Complete rewrite of core internals
- Modularization of each component allowing easier implementation of integration methods
- Error reporting major improvment. Less random exits with no warnings. Better exits.
- Focus on object orientation for improved library access
- Speed up of matrix creation algorithm
- Improvements to output format
- Inclusion of '.include' control card
- Capability to read from standard in using '-i' command
- Various bug fixes

### v2.3 - 10/05/2019
- Major changes to JoSIM internal engine
- Removal of plotting engines in favor of external tools such as Python
- Introduction of pyJoSIM and libjosim packages
- Immense speed-up of internals
- Various bug fixes

### v2.2 - 12/02/2019
- Major rewrite of most of JoSIM code to accommodate library creation for use in C++ and Python
- Implementation of a phase source for use in phase-mode simulation

### v2.1 - 16/01/2019
- Inclusion of the Haberkorn junction current which allows for user switching between ballistic and normal electron tunneling
- Various bug fixes

### v2.0 - 10/10/2018
- New analysis method that allows Modified Nodal Phase Analysis (see ReadMe)
- Major memory improvement by not creating intermediate A matrix
- Various bug fixes

### v1.3 - 28/07/2018
- Addition of mutual inductance with the netlist label **K**
- Addition of the sinusoidal source (Voltage and Current)
- Various bug fixes
- AQFP now works

### v1.2 - 17/07/2018
- New Release brings RType=1 to the RCSJ model of the Josephson Junction.
- Various bugs have been fixed. 
- Data structures have been majorly altered.
- Other changes and fixes

### v1.1 - 04/06/2018
- Ability to change between plotting libraries (NONE, FLTK or Matplotlib [Default])
- Major bug fixes
- JoSIM now supports WRSpice syntax (mostly)
- Parameterization of component values with expression parsing
- Code cleanup and commenting

### v1.0 - 03/05/2018
- Initial release of JoSIM
- Default FLTK plotting engine
- Replaces JSIM as default simulator
