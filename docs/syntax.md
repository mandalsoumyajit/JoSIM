# Syntax Guide

In this section we will attempt to provide the user with a comprehensive guide of the available syntax within JoSIM

JoSIM is ***CaSe InSeNsItIvE*** as each line is cast to uppercase upon read-in.

Additionally, it is **discouraged** to add units to values since in some specific cases this creates confusion in interpretation of the values. An example of this is specifying Farad when assigning a capacitor its value since a value followed by F would be interpreted as Femto. This will lead to wanting a 5 Farad capacitor but receiving a 5 *Femto* Farad capacitor. Units are not interpreted by the simulator in any way since the component type implies the unit. 

## Basic Syntax

Each line follows similar syntax which uses the first non-blank space character as identifier. Each identifier tells JoSIM how to handle that specific line. 

Identifiers that start with a letter relate to physical components in the design, e.g. L, C, R. Lines of this kind almost always follows the same syntax in that it requires a label and two nodes. These nodes can be alphanumeric with the restriction of **0** and **GND** which indicate a grounded node. Additionally, the use of period (**.**) or vertical bars (**|**) in label or node names are prohibited as these are reserved characters within JoSIM.

Lines that start with a period (**.**) indicate that the line relates in some way to simulation control. In this case the command that follows the period identifies the control, e.g. .tran, .print, .end.

Comments are lines that start with an asterisk (**\***) or a hash (**\#**). Comments are meant to be in a line of their own and will not work if placed at the end of a line.

Lines that start with a plus sign (**+**) indicate that the line is a continuation of the previous line. Internally the two lines will be combined.

In most cases the **VALUE** of a component can be replaced by a variable name or an expression. Variables can be defined using the **.PARAM** control. These will be discussed in detail further.

Values in JoSIM can be modified with engineering notation or through suffixes. A list of the available suffixes is found below:

| Suffix | Meaning | Engineering Notation Equivalent |
| ------ | ------- | ------------------------------- |
| F      | Femto   | 1E-15                           |
| P      | Pico    | 1E-12                           |
| N      | Nano    | 1E-9                            |
| U      | Micro   | 1E-6                            |
| M      | Milli   | 1E-3                            |
| K      | Kilo    | 1E3                             |
| MEG    | Mega    | 1E6                             |
| X      | Mega    | 1E6                             |
| G      | Giga    | 1E9                             |
| T      | Tera    | 1E12                            |

## Basic Components

We will now run through all the available physical components and their limitations. Any parameter surrounded by square brackets are optional and nested square brackets mean that the encapsulated parameter relies on the existence of the previous.

### Resistor

**R**Label&emsp;$N^{+}$&emsp;$N^{-}$&emsp;**VALUE**&emsp;[temp=<**TEMP**>]&emsp;[neb=<**FREQ**>]

The value of a resistor is in Ohms.

Temperature used for noise analysis in  Kelvin. Units need to be excluded since **K** for Kelvin will be interpreted as Kilo by JoSIM, e.g. 4K will become 4000.

Optional frequency parameter sets the noise effective bandwidth during noise analysis. Default is 1GHz.

### Inductor

**L**Label&emsp;$N^{+}$&emsp;$N^{-}$&emsp;**VALUE**

The value of an inductor is in Henry.

### Capacitor

**C**Label&emsp;$N^{+}$&emsp;$N^{-}$&emsp;**VALUE**

The value of a capacitor is in Farad.

### Josephson Junction (JJ)

**B**Label&emsp;$N^{+}$&emsp;$N^{-}$&emsp;*<PhaseNode\>*&emsp;**MODEL**&emsp;[area=<**AREA**\>]&emsp;[ic=<**IC**>]&emsp;[temp=<**TEMP**>]&emsp;[neb=<**FREQ**>]

A Josephson junction is a two terminal device but could also be defined with a third non-connected node to allow compatibility with WRspice. This node is not used in any way in JoSIM.

The Josephson junction requires specification of a model name which can be defined anywhere in the program using the control **.MODEL**.

When **AREA** or **IC** is not specified then an area=1 is used as default.

The temp and neb commands have the same descriptions as for the resistor.

#### Model

This model control has the following syntax

**.MODEL**&emsp;*ModelName*&emsp;*ModelType*([**MODEL PARAMETERS**])

The only junction model currently supported by JoSIM is the RCSJ model and thus the only available ModelType is **jj** with the following tunable parameters:

| Parameter       | Range                 | Default         | Description                                                  |
| --------------- | --------------------- | --------------- | ------------------------------------------------------------ |
| RTYPE           | 0, 1                  | 1               | Linearization model used                                     |
| VG or VGAP      | -$\infty$, $\infty$   | 2.8E-3          | Junction gap voltage                                         |
| IC or ICRIT     | -$\infty$, $\infty$   | 1E-3            | Junction critical current                                    |
| RN              | 0, $\infty$           | 5               | Junction normal resistance                                   |
| R0              | 0, $\infty$           | 30              | Junction subgap resistance                                   |
| C or CAP        | 0, $\infty$           | 2.5E-12         | Junction capacitance                                         |
| T               | 0, $\infty$           | 4.2             | Junction temperature in K                                    |
| TC              | 0, $\infty$           | 9.1             | Critical temperature of superconducting material             |
| DELV            | 0, $\infty$           | 0.1E-3          | Transitional voltage from subgap to normal                   |
| D               | 0.0, 1                | 0.0             | Point of contact transparency affecting current phase relationship |
| ICFCT or ICFACT | 0, 1                  | $\frac{\pi}{4}$ | Ratio of critical current to quasiparticle step height       |
| PHI             | 0, $2\pi$             | 0               | Allows phi-junction capability such as the $\pi$-junction.   |
| CPR             | {$-\infty$, $\infty$} | {1}             | Sets the Current Phase Relationship harmonic amplitudes.     |

The *.model* line is unique to the subcircuit it falls under and can thus allow different models with the same name under separate subcircuits. If the model is not found under the subcircuit it will be searched for globally and if not found default values (default model) will be used instead.

The **AREA** and **IC** parameters act as modifiers to the model parameters. **AREA** is a critical current multiplier, where if **IC** is specified it replaces the **AREA** value by $AREA=\frac{IC_{jj}}{IC_{model}}$. 

By setting the **PHI** parameter of the model, the phase value is persistantly subtracted from the phase ($\phi$) in the $\sin(\phi)$ part of the JJ current. This allows elements such as the $\pi$-junction to be modeled. 

The **CPR** parameter alters each individual harmonic of the junction supercurrent. It can either be a singular value or an array of *n* values encapsulated with curly braces. i.e *CPR={0.9, 0.1}* to set the current phase relationship to the following $I = I_C(0.9\sin(\phi) + 0.1\sin(2\phi))$. 

### Transmission Line

**T**Label&emsp;$N^{+}_{1}$&emsp;$N^{-}_{1}$&emsp;$N^{+}_{2}$&emsp;$N^{-}_{2}$&emsp;**TD=VALUE**&emsp;**Z0=VALUE**&emsp;[interp=<**0|1**>]

**TD** is the time delay in seconds.

**Z0** is the impedance in Ohms.

**INTERP** (optional, `1`, `TRUE`, or `YES`) enables linear fractional-step interpolation on this transmission line when **TD** is not an integer multiple of the simulation timestep $T_{STEP}$. Fractional delay interpolation can also be enabled globally for all transmission lines in the netlist using `.option txinterp` (or `.option fractx`). When disabled (the default), $\text{TD} / T_{STEP}$ is rounded to the nearest integer step count.

### Mutual Inductance

**K**Label&emsp;$L_{1}$&emsp;$L_{2}$&emsp;**VALUE**

The value is the coupling factor *k*.

## Sources

### Independent Sources

#### Voltage Source

**V**Label&emsp;$N^{+}$&emsp;$N^{-}$&emsp;**SOURCETYPE**

#### Current Source

**I**Label&emsp;$N^{+}$&emsp;$N^{-}$&emsp;**SOURCETYPE**

#### Phase Source

**P**Label&emsp;$N^{+}$&emsp;$N^{-}$&emsp;**SOURCETYPE**

### Dependent Sources

#### Current controlled current source

**F**Label&emsp;$N^{+}$&emsp;$N^{-}$&emsp;$N^{+}$Control&emsp;$N^{-}$Control&emsp;**CURRENT GAIN**

#### Current controlled voltage source

**H**Label&emsp;$N^{+}$&emsp;$N^{-}$&emsp;$N^{+}$Control&emsp;$N^{-}$Control&emsp;**TRANSRESISTANCE IN OHMS**

#### Voltage controlled current source

**G**Label&emsp;$N^{+}$&emsp;$N^{-}$&emsp;$N^{+}$Control&emsp;$N^{-}$Control&emsp;**TRANSCONDUCTANCE IN MHOS**

#### Voltage controlled voltage source

**E**Label&emsp;$N^{+}$&emsp;$N^{-}$&emsp;$N^{+}$Control&emsp;$N^{-}$Control&emsp;**VOLTAGE GAIN**

### Source Types

#### Piece Wise Linear (PWL)

**pwl(0**&emsp;**0**&emsp;$T_{1}$&emsp;$A_{1}$&emsp;...&emsp;$T_{n}$&emsp;$A_{n}$**)**

This source linearly interpolates amplitude values for every time point in the simulation between the specified amplitudes.

The initial two values are required to be zero at the start of the simulation.

#### Pulse

**pulse(**$A_{1}$&emsp;$A_{2}$&emsp;[$T_{D}$&emsp;[$T_{R}$&emsp;[$T_{F}$&emsp;[*PW*&emsp;[*PER*]]]]] **)**

This source generates a pulse between two amplitudes ($A_{1}$ and $A_{2}$), starts after $T_{D}$ and has a rise and fall time ($T_{R}$ & $T_{F}$), which default to the transient simulation step size.

*PW* and *PER* refer to the pulse width and the period respectively. These values default to the transient simulation stop time when not specified.

This source allows the continuous generation of a pulse at a set frequency.

#### Sinusoidal

**sin(**$A_{O}$&emsp;$A$&emsp;[$f$&emsp;[$T_{D}$&emsp;[$\theta$&emsp;[$\phi$&emsp;[$N_{C}$]]]]] **)**

A source that generates a (optionally damped) sinusoidal signal with DC offset $A_{O}$ and amplitude $A$ at frequency $f$ (Hz). If omitted, $f$ defaults to $\frac{1}{T_{STOP}}$.

$T_{D}$ is a start delay: for $t<T_{D}$ the output is held at $A_{O}$.

$\theta$ is an exponential damping factor (units $s^{-1}$). When $\theta=0$, no damping is applied.

$\phi$ is a phase shift applied to the sine term. Internally it is treated as radians; if specified in degrees it is converted via $\phi=\phi_{\deg}\pi/180$.

If $N_{C}$ is provided and $N_{C}$ $>0$, the sinusoid runs for that many cycles and then stops, returning to the offset $A_{O}$ for all subsequent times.

The function generates a data point for each step in the transient simulation based on the following piecewise definition (let $\tau=t-T_{D}$):

$$
f(t)=
\begin{cases}
A_{O}, & t < T_{D}\\[4pt]
A_{O} + A\,e^{-\theta \tau}\,\sin\!\big(2\pi f\,\tau + \phi\big), &
T_{D} \le t < T_{D}+\dfrac{\text{$N_{C}$}}{f}\quad(\text{when $N_{C}$}>0)\\[10pt]
A_{O}, & t \ge T_{D}+\dfrac{\text{$N_{C}$}}{f}\quad(\text{when $N_{C}$}>0)\\[6pt]
A_{O} + A\,e^{-\theta \tau}\,\sin\!\big(2\pi f\,\tau + \phi\big), &
t \ge T_{D}\quad(\text{when $N_{C}$}\le 0\ \text{or omitted})
\end{cases}
$$

Notes:

- If $N_{C}$ is omitted (or $\le 0$), the sinusoid continues for the full simulation duration.
- If $N_{C}$ $>0$, $f$ must be positive to define the stop time.

#### Custom Waveform

**cus(** *wavefile*&emsp;$T_{S}$&emsp;*SF*&emsp;*IM*&emsp;[$T_{D}$&emsp;*PER]* **)**

This source allows the generates a function based on the points inside the plain text wave file. This file should contain a single line of space separated numbers. E.g 0 2 3 6 2 1 0

Each number in this line represents an amplitude separated by time step $T_{S}$ and scaled using scale factor *SF*. The values between the points are interpolated using either no interpolation (0), linear (1) or cubic (2). The function can become periodic if PER is set to 1, whereby the pattern is repeated for the entire simulation.

The waveform only starts of $T_{D}$.

#### DC

*dc*&emsp;$A$

A DC source that is always at $A$ at any given time during the simulation.

#### Noise

**noise(**$A$&emsp;$T_{D}$&emsp;$T_{STEP}$**)**

This source produces a noise value for the time step provided that it is after $T_{D}$. The $T_{STEP}$ is the inverse of the noise effective bandwidth set globally using `.neb` or locally for a resistor using `neb=`.

The noise value returned is calculated using the following equation:

$f(t) = A\frac{GRAND()}{\sqrt{2T_{STEP}}}$

$GRAND()$ is a Gaussian random number generating function.

#### Exponential

**exp(** $A_{1}$&emsp;$A_{2}$&emsp;$T_{D1}$&emsp;$\tau_{1}$&emsp;$T_{D2}$&emsp;$\tau_{2}$**)**


| Parameter  | Default           |
| ---------- | ----------------- |
| $T_{D1}$   | 0.0               |
| $T_{D2}$   | $T_{STEP}$        |
| $\tau_{1}$ | $T_{D1}+T_{STEP}$ |
| $\tau_{2}$ | $T_{STEP}$        |


Returns different values for the 3 different time segments.

For $t < T_{D1}$:

$f(t)=A_{1}$ 

For $T_{D1}\le t < T_{D2}$:

$f(t) = A_{1}+(A_{2}-A_{1})(1-e^{\frac{t - T_{D1}}{\tau_{1}}})$

For $T_{D2} \le t$:

$f(t) = A_{1}+(A_{2}-A_{1})(1-e^{\frac{t - T_{D1}}{\tau_{1}}})+(A_{1}-A_{2})(1-e^{\frac{t - T_{D2}}{\tau_{2}}})$

#### Piecewise Sinusoidal (PWS)

**pws(**$T_{0}$&emsp;$V_{0}$&emsp;$T_{1}$&emsp;$V_{1}$&emsp;...&emsp;$T_{N}$&emsp;$V_{N}$ **)**

A source that generates a smooth, sinusoidally-interpolated waveform through a series of user-defined time/value points, similar to **pwl(...)** but using a sinusoidal spline between each adjacent pair of points instead of linear interpolation.

Each point is specified as a time/value pair \((T_i, V_i)\). Time values **must be strictly increasing**.

- For \(t < T_0\), the output is held at \(V_0\).
- For \(t \ge T_N\), the output is held at \(V_N\).
- For \(T_i \le t < T_{i+1}\), the output transitions smoothly from \(V_i\) to \(V_{i+1}\) using a half-cosine (sinusoidal easing) segment.

Let \(u=\dfrac{t-T_i}{T_{i+1}-T_i}\) for the segment \(i\), where \(0 \le u < 1\). The output is:

\[
f(t)=V_i + (V_{i+1}-V_i)\,\frac{1-\cos(\pi u)}{2},
\qquad T_i \le t < T_{i+1}
\]

This interpolation is continuous and has zero slope at each specified point, producing smooth joins at breakpoints.

Notes:

- The final value is held constant after the last timestep \(T_N\).
- Duplicate or non-increasing time values are invalid and may result in errors.


## Control Commands

The simulation engine requires control commands to know what to do with the components it has just been provided with.

The most important of these control commands is the transient simulation command as no simulation can be performed without it.

### Transient Analysis

**.tran**&emsp;$T_{STEP}$&emsp;$T_{STOP}$&emsp;[$P_{START}$&emsp;[$P_{STEP}$&emsp;[$P_{WINDOW}$]]]&emsp;DST

This generates a simulation that runs from 0 until $T_{STOP}$. The amount simulation steps that will be performed is $n=\frac{T_{STOP}}{T_{STEP}}$.

$P_{START}$ indicates at what point output will start printing. $P_{STEP}$ sets the size of the print steps. This has to be larger or equal to $T_{STEP}$.

If $P_{WINDOW}$ is specified and not 0, a Hanning FIR filter with a window width of $P_{WINDOW}$ seconds is applied to the output data before decimating as a post-processing step. This can be used to low-pass filter the output to prevent aliasing when decimating with $P_{STEP}$, e. g. to create an IV curve of the whole circuit.

DST disables the start-up time. The start-up time is a period calculated internally by the simulator in which components settle. This is equivalent to the few picoseconds from when a circuit initially receives power (power switch flipped).

### Subcircuits

Subcircuits allow subdivision and reuse of smaller circuits within a larger design. When wrapped in a subcircuit control devices are allowed to have the same label names as specified elsewhere in the netlist as the subcircuit completely isolates them.

Subcircuits have the following wrapping control syntax

**.subckt**&emsp;*SubcktName*&emsp;*IO Nodes*

...

**.ends**

The *SubcktName* specifies the name of the subcircuit and *IO Nodes* specify the which nodes within the subcircuit connects to outside.

A subcircuit can be used in the main netlist or another subcircuit (nesting) using the following syntax

**X**Label&emsp;*SubcktName*&emsp;*IO Nodes*&emsp;(JSIM mode)

**X**Label&emsp;*IO Nodes*&emsp;*SubcktName*&emsp;(WRspice (normal SPICE) mode)

Additionally, keywords in the form of **LABEL=VALUE** can be appended to the end of the subcircuit declaration line which when instantiated will replace the value of the **LABEL** component within the subcircuit with the associated **VALUE**. This allows for unique subcircuit instantiations which would prove useful in testing various parameters without altering the original subcircuit or having multiple instances of the same subcircuit definition. This could open the door for potential future margin and optimization software.

### Noise

As mentioned in the technical discussion, noise can be automatically inserted as current sources in parallel to each resistor. This thermal noise temperature and bandwidth can be specified globally using the following commands:

**.temp**&emsp;*Temperature in Kelvin*

**.neb**&emsp;*Bandwidth in Hertz*

### Spread

JoSIM allows each value of inductors, resistors, capacitors and JJ (area/Ic) to be spread uniformly within a specified percentage range from the nominal value. Each time the value is used (in matrix creation), a new random value from the uniform distribution is chosen.

This allows for process variation to be simulated.

To set the spread globally, the following control needs to be set:

**.spread**&emsp;*Normalized percentage*&emsp;*[L=Inductor spread]*&emsp;*[B=JJ spread]*&emsp;*[C=Capacitor spread]*&emsp;*[R=Resistor spread]*

Each individual component can also be spread by adding the **spread=** named parameter to the component declaration line.

The order of precedence is taken as local, specific global and then global. This means that if a global spread exists but  a global inductor specific spread also exists and the inductor being added has a local spread then the local spread will take precedence.

An example:

If an inductor has a value of *2pH*, this is its nominal value. If a local spread of *0.2* is specified this means a random value can be chosen anywhere between *1.6pH* and *2.4pH*. If global inductor specific spread is specified as *0.3* and the local spread is not defined, the value can be anything within the range *1.4pH -  2.6pH*. Similarly if the global spread is defined as *0.5* then the value each time the simulation is run can be anywhere between *1pH* and *3pH*.

### Options and Reproducibility (`.option`)

Simulator options can be configured in the netlist using the `.option` (or `.options`) control card:

**.option**&emsp;*OPTION_KEY*[=*VALUE*]&emsp;[*OPTION_KEY2*[=*VALUE2*] ...]

| Option | Syntax | Description |
| ------ | ------ | ----------- |
| **SEED** | `.option seed=<int>` | Sets a deterministic integer seed for the random number generator used by **noise** and **spread**. The current run's seed is also printed in verbose mode (`-V 1`). |
| **TXINTERP** / **FRACTX** | `.option txinterp` | Enables linear fractional-step delay interpolation for all transmission lines (`T` elements) when $T_D / T_{STEP}$ is non-integer. Can also be set per element via `interp=1`. |
| **FIXEDSTEP** | `.option fixedstep` | Disables automatic global timestep halving (`reduce_step`) when a Josephson junction's phase step exceeds $0.4\pi$. Equivalent to the `-f` (`--fixed-step`) CLI switch. |
| **WARNSTEP** | `.option warnstep` | Prints a diagnostic warning identifying the offending Josephson junction label, halving count, and old/new `TSTEP` whenever automatic step halving or a phase jump occurs. |
| **CORRECTOR** | `.option corrector=<N>` | Enables up to `N` implicit predictor-corrector (Picard/BDF2) iterations per timestep on active Josephson junctions (default `0`; `-c` on the CLI defaults to `2`). |
| **CORRECTORTOL** | `.option correctortol=<val>` | Sets the phase convergence threshold (in radians, default `1e-3`) that triggers predictor-corrector iteration when `CORRECTOR` > 0. |
| **KCUTOFF** / **KMIN** | `.option kcutoff=<val>` | Ignores mutual inductance entries (`K` lines) with $|k| < \text{val}$ during MNA matrix construction, preventing dense LU fill-in from weak parasitic couplings in 3D field-extracted netlists. |

Examples:
```cir
.option seed=4190754512324517577
.option txinterp warnstep
.option corrector=2 correctortol=1e-4
.option kcutoff=1e-3
```

### State Checkpointing (`.savestate` / `.loadstate`)

For large circuits that require a long initial bias-conditioning or settling window before applying data stimuli, JoSIM can serialize and restore the full simulator state (the MNA solution vector $\mathbf{x}$, all multi-step device history registers, and the complete circular delay ring buffers of all transmission lines) in a compact binary checkpoint file (`JOSIMSTA` v2):

**.savestate**&emsp;*filepath*&emsp;[*save_time*]

**.loadstate**&emsp;*filepath*

- **.savestate** (or alias **.savebias**) writes the binary checkpoint to *filepath*. If *save_time* is specified, the state is captured at the first timestep where $t \ge \text{save\_time}$; otherwise, it is written at the end of the transient simulation ($T_{STOP}$).
- **.loadstate** (or alias **.loadbias**) loads the binary checkpoint from *filepath* at the start of the transient simulation ($t = 0$), restoring all node voltages/phases, branch currents, Josephson junction state variables, and transmission-line delay buffers.

Example:
```cir
* Run 1: Save conditioned bias state at 6000 ps
.savestate bias_6ns.state 6000p

* Run 2: Start immediately from the conditioned bias state
.loadstate bias_6ns.state
```

### IV Curve

JoSIM allows the user to output an IV curve for a specified JJ model within the netlist using the following command:

**.iv**&emsp;*modelname*&emsp;*max_current*&emsp;*filepath*

This command outputs a comma seperated value (CSV) at the *filepath* specified which contains the IV curve data for the *modelname* from negative *max_current* to positive *max_current*.

Subcircuit models can be output using the `.`(period) or `|`(vertical bar) as separator between the *modelname* and the subcircuit NAME.

### Output

A simulation is meaningless unless the results are post processed. In order to know which of these results are relevant for storage the simulator needs output control commands.

These output commands can be of either 

**.print**&emsp;**.plot**&emsp;**.save**

Any of these commands can be followed by either of the following commands

*PrintType*&emsp;&nbsp;*Device* or *Node*

*PType(Device or Node)*\(_0\)&emsp;&nbsp;*...*&emsp;*PType(Device or Node)*\(_n\)

Where *PrintType* can be either device voltage (*DEVV*), device current (*DEVI*), device phase (*PHASE*), node voltage (*NODEV*) or node phase (*NODEP*).

When specifying a device type store only a single device can be specified, but when a node type store is specified 2 nodes can be specified to store the difference between them.

*PType* is shorthand for the above and can have multiple per line requests. *PType* can be either of *V*, *I* (or *C*) or *P* followed by the device or node in brackets. If more than one device or node is specified by comma seperation (maximum 2) the difference between the two devices or nodes is stored.

An additional save type exists that has the following syntax

@*Device*[*PType*]

This stores the *PType* of the device specified.

Subcircuit parameters can be output using the `.`(period) or `|`(vertical bar) as separator between the device label and the subcircuit label name. 

i.e. **.print** v(14.X01) p(B01.X02)

This method follows the path from inside to out when nested subcircuits are used. 

Subcircuits as mentioned before can also be nested almost indefinitely as they are expanded upon simulation.

If the label is not immediately apparent and required for output, the simulation can be run using the `-V 3` cli option to show the expanded main design, allowing the exact label name to be identified. 

### File

Multi-file output can be achieved by separating output commands with the following command:

**.file**&emsp;*filepath*

This indicates that every output request that follows this command needs to be stored in the file specified by *filepath*. Both `.file` and the `-o` (`--output=`) command-line switch automatically select the output format from the file extension:

| Extension | Format | Description |
| --------- | ------ | ----------- |
| `.csv` | Comma-Separated Values (ASCII) | Standard comma-separated text table with header row (`time,...`). |
| `.dat` | Space-Separated Values (ASCII) | Standard space-separated text table with header row. |
| `.raw` or no extension | SPICE3 Raw (ASCII) | Standard ASCII SPICE rawfile (`Values:` section). |
| `.bin` | Native JoSIM Binary (`JOSIMBIN`) | High-speed column-major IEEE-754 `float64` binary format with self-describing trace metadata header. ~37x faster to write and ~38% smaller than `.csv`. |
| `.npy` | NumPy Structured Array (`.npy`) | Standard NumPy v1.0/v2.0 structured array with named `<f8` columns, directly loadable via `numpy.load()` or `scripts/josim_io.py`. |
| `.rawb` or `.braw` | SPICE3 Binary Raw (`Binary:`) | Standard SPICE3 binary rawfile with ASCII preamble followed by interleaved IEEE-754 `float64` records. |

I.e. 

```
.file output1.csv
.print p(B1) p(B2) p(B3)
.file output2.dat
.plot i(L01) i(L02) i(L03)
.file output3
.save v(1) v(2) v(3)
.file output4.bin
.print v(1) i(L01) p(B1)
.file output5.npy
.print v(1) i(L01) p(B1)
```

This will create 5 output files: a CSV file containing the phase outputs, a space-separated `.dat` file containing the currents, an ASCII SPICE `.raw` file containing the voltages, a native column-major `.bin` binary file, and a NumPy `.npy` structured array file.

Binary `.bin` and `.npy` files can be loaded in Python using [`scripts/josim_io.py`](https://github.com/JoeyDelp/JoSIM/blob/master/scripts/josim_io.py):
```python
from scripts.josim_io import load_josim
data = load_josim("output4.bin")  # dict mapping trace name -> 1D numpy float64 array
```

The use of this command does not affect the command line option request to output a file. The command line output option, if requested, will output an additional file which contains all of the output requests (phase, current and voltage).

### Parameters

The final control command that is of importance in JoSIM is the parameters command with the following syntax

**.param** &emsp;*VarName*=*Expression*

*VarName* is the variable name that can be used anywhere else in the circuit and *Expression* is a mathematical expression that is evaluated using an implementation of Dijkstra's shunting yard algorithm, whereby the expression is converted into reverse polish notation (RPN) and evaluated.

Additionally, expressions can also contain other variables and parameters will be continuously evaluated until all variables are reduced to values. If variables are not defined the program will halt and produce an error.

Expression parsing is exclusive to the *.param* control. This means that if expressions are loosely provided as values to components or as parameters to plot or model controls, JoSIM will error in *std::invalid_argument: stod: no conversion* as it tries to convert a string into a double. Please be mindful when using expressions and restrict them to *.param* controls.

### Control Block

Any of the above controls can be wrapped inside a control block with the following syntax

**.control**

...

**.endc**

Wherein all controls can be specified by omitting the usual prepending `.`(period) to the command.

This block, though seemingly useless at present, will be used for more advanced functionality in the future.

### Include

JoSIM allows the use of a *.include* control card that uses the following syntax

**.include** *relative_path_to_file*

This command reads in the contents of the relevant file pointed to by the relevant path upon parsing of the netlist essentially extending the netlist by the linked file. This is incredibly handy when large subcircuits are involved and reuse of subcircuits across multiple files is required.

This can also be used to house all the models used in simulation allowing a central point of alteration if the model is changed.

### Standard Input

JoSIM now allows input from standard input allowing a line-by-line read in of a netlist until the *.end* card is found or alternatively the EOF character is returned.



## Constants

JoSIM has a set of built in constants that when used expand to the corresponding values. Below is a list of these constants and their values. The constants names are case insensitive:

| Constant  | Symbol                                       | Value                   |
| --------- | -------------------------------------------- | ----------------------- |
| PI        | $\pi$                                        | 3.141592653589793238463 |
| PHI_ZERO  | $\Phi_{0}$                                   | 2.067833831170082E-15   |
| BOLTZMANN | $k_{B}$                                      | 1.38064852E-23          |
| EV        | $e$                                          | 1.6021766208E-19        |
| HBAR      | $\bar{h}$                                    | 1.0545718001391127E-34  |
| C         | $c$                                          | 299792458               |
| MU0       | $\mu_{0}$                                    | 12.566370614E-7         |
| EPS0      | $\epsilon_{0}$                               | 8.854187817E-12         |
| SIGMA     | $\sigma$ (short for $\frac{\Phi_{0}}{2\pi}$) | 3.291059757E-16         |

[^1]: I. Salameh, E. G. Friedman and S. Kvatinsky, "Superconductive Logic Using 2ϕ—Josephson Junctions With Half Flux Quantum Pulses," in *IEEE Transactions on Circuits and Systems II: Express Briefs*, vol. 69, no. 5, pp. 2533-2537, May 2022, doi: 10.1109/TCSII.2022.3162723.
[^2]:S. V. Bakurskiy et al., "Current-phase relations in SIsFS junctions in the vicinity of 0-$\pi$ Transition", *Phys. Rev. B Condens. Matter*, vol. 95, pp. 94522-94528, Mar. 2017.
