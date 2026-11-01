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

**L**Label&emsp;$N^{+}$&emsp;$N^{-}$&emsp;**VALUE**&emsp;[flux=<**PHI0_FRAC**> | phi0=<**PHI0_FRAC**> | phi=<**RAD**>]&emsp;[ic=<**INITIAL_CURRENT**>]&emsp;[spread=<**SPREAD**>]

The value of an inductor is in Henry.

Optional static flux-bias and initial-condition modifiers (*[New in v2.8 Fork]*):
- **`flux=<val>`** or **`phi0=<val>`**: Applies a static external magnetic flux offset across the inductor in units of $\Phi_0$ ($\Delta\phi_{\text{off}} = 2\pi \cdot \text{val}$ radians, equivalent to trapped flux or an inductively coupled DC flux bias without adding extra current source or `K` branches).
- **`phi=<val>`**: Applies a static phase offset directly in radians.
- **`ic=<val>`**: Sets an initial inductor branch current (in Amperes) at $t = 0$.

### Capacitor

**C**Label&emsp;$N^{+}$&emsp;$N^{-}$&emsp;**VALUE**&emsp;[ic=<**INITIAL_VOLTAGE**>]&emsp;[spread=<**SPREAD**>]

The value of a capacitor is in Farad. If **`ic=<val>`** is provided, the capacitor voltage is initialized to `<val>` Volts at $t = 0$.

### Josephson Junction (JJ)

**B**Label&emsp;$N^{+}$&emsp;$N^{-}$&emsp;*<PhaseNode\>*&emsp;**MODEL**&emsp;[area=<**AREA**\>]&emsp;[ic=<**IC**>]&emsp;[phi0=<**INIT_PHASE_RAD**>]&emsp;[phioff=<**PHASE_OFFSET_RAD**>]&emsp;[temp=<**TEMP**>]&emsp;[neb=<**FREQ**>]&emsp;[spread=<**SPREAD**>]

A Josephson junction is a two terminal device but could also be defined with a third non-connected node to allow compatibility with WRspice. This node is not used in any way in JoSIM.

The Josephson junction requires specification of a model name which can be defined anywhere in the program using the control **.MODEL**.

When **AREA** or **IC** is not specified then an area=1 is used as default.

The temp and neb commands have the same descriptions as for the resistor. If **`phi0=<rad>`** is specified, the junction phase starts at `<rad>` radians at $t = 0$; **`phioff=<rad>`** adds a per-instance static phase offset (in radians) on top of the model's `PHI` parameter.

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

### Superconducting Nanowire / Cryotron (`N` — `nTron`, `hTron`, `SNSPD`) — *[New in v2.8 Fork]*

**N**Label&emsp;$N_{\text{drain}}$&emsp;$N_{\text{source}}$&emsp;[$N_{\text{gate}+}$&emsp;[$N_{\text{gate}-}$]]&emsp;**MODEL**&emsp;[area=<**AREA**>]&emsp;[ic=<**IC0**>]&emsp;[ig=<**IG0**>]&emsp;[ir=<**IR**>]&emsp;[rn=<**RN**>]&emsp;[l=<**LKIN**>]&emsp;[lg=<**LG**>]&emsp;[rg=<**RG**>]&emsp;[tau=<**TAU_TH**>]&emsp;[tau_hs=<**TAU_HS**>]&emsp;[alpha=<**ALPHA**>]&emsp;[gate=<**BRANCH_OR_ISRC**>]

Models a 2-terminal superconducting nanowire / SNSPD, a 3-terminal nanocryotron (`nTron`, where the gate return is ground `0`), or a 4-terminal heater/isolated-gate cryotron (`hTron`).

**.MODEL**&emsp;*ModelName*&emsp;**ntron | htron | snspd | nanowire**([**MODEL PARAMETERS**])

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `IC`, `ICRIT`, `IC0` | `100u` | Zero-gate channel switching current $I_{c0}$ (A). |
| `IG`, `IGSW`, `IG0` | `20u` | Gate critical current $I_{g0}$ that suppresses channel $I_{\text{sw}}$ to zero (A). |
| `IR`, `IRET`, `IHS` | $0.2 I_{c0}$ | Retrapping / hotspot collapse current $I_r$ (A). |
| `RN`, `RHS`, `RNORM` | `500` | Normal-state hotspot channel resistance $R_n$ ($\Omega$). Alternatively computed from `RSHEET * (LCH / WCH)`. |
| `L`, `LK`, `LKIN`, `LKIN0` | `40p` | Channel kinetic inductance $L_k$ (H). |
| `LG`, `LGATE` | `5p` | Synthesized gate-branch inductance (H, for 3-/4-terminal devices). |
| `RG`, `RGATE` | `0.0` | Synthesized gate-branch series resistance ($\Omega$). |
| `TAU`, `TAUTH`, `TAU_TH` | `15p` | Thermal recovery (cooling) time constant $\tau_{\text{th}}$ (s). |
| `TAUHS`, `TAU_HS` | `2p` | Hotspot nucleation / growth time constant $\tau_{\text{hs}}$ (s). |
| `ALPHA`, `EXP` | `1.0` | Gate suppression exponent $\alpha$ in $I_{\text{sw}}(I_g) = I_{c0}\max(0, 1 - (|I_g|/I_{g0})^\alpha)$. |

Probing: `V(N1)`, `I(N1)`, `P(N1)` probe the nanowire channel; `I(N1|G)` or `I(N1|GATE)` probes the synthesized gate branch.

### Transmission Line

**T**Label&emsp;$N^{+}_{1}$&emsp;$N^{-}_{1}$&emsp;$N^{+}_{2}$&emsp;$N^{-}_{2}$&emsp;**TD=VALUE**&emsp;**Z0=VALUE**&emsp;[interp=<**0|1**>]&emsp;[db=<**DB**> | loss=<**FRAC**> | atten=<**GAMMA**> | alpha=<**ALPHA**> [len=<**L**>]]

**TD** is the time delay in seconds.

**Z0** is the impedance in Ohms.

**INTERP** (optional, `1`, `TRUE`, or `YES`) enables linear fractional-step interpolation on this transmission line when **TD** is not an integer multiple of the simulation timestep $T_{STEP}$. Fractional delay interpolation can also be enabled globally for all transmission lines in the netlist using `.option txinterp` (or `.option fractx`). When disabled (the default), $\text{TD} / T_{STEP}$ is rounded to the nearest integer step count.

**Lossy Wave Attenuation Parameters** (*[New in v2.8 Fork]*):
- **`db=<val>`** (or `loss_db=<val>`, `atten_db=<val>`): One-way wave attenuation in decibels ($\gamma = 10^{-|\text{val}|/20}$).
- **`loss=<frac>`**: Fractional one-way amplitude loss ($\gamma = 1 - \text{frac}$).
- **`atten=<val>`** (or `gamma=<val>`, `gain=<val>`): Direct one-way wave transmission factor ($\gamma \in [0, 1]$).
- **`alpha=<val>` [with optional `len=<L>` or `nl=<L>`]**: Exponential attenuation ($\gamma = e^{-|\alpha| L}$ if length is specified, or $\gamma = e^{-|\alpha| T_d}$ in $\text{Np/s}$).

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

#### Digital Bit-Stream Pattern (`PAT`) — *[New in v2.8 Fork]*

**pat(**$V_0$&emsp;$V_1$&emsp;$T_D$&emsp;$T_R$&emsp;$T_F$&emsp;[*PW*]&emsp;*PER*&emsp;*PATTERN*&emsp;[RZ | NRZ]&emsp;[r=<**REPEAT**>]&emsp;[rb=<**START_BIT**>]**)**

Generates a Return-to-Zero (`RZ`, when `PW < PER` or `RZ` is specified) or Non-Return-to-Zero (`NRZ`, when `PW` is omitted or `NRZ` is specified) digital bit-stream waveform between low level $V_0$ and high level $V_1$:
- **PATTERN**: Binary/ternary string (`10110100`, `10-1`, `B101101`) or hexadecimal string (`0xB4`, `HEX:A5`). Ternary `-` or `N` pulses to $2V_0 - V_1$ (negative amplitude).
- **`r=<N>`**: Number of times to repeat the pattern (`-1` or `INF` repeats indefinitely; default `1`).
- **`rb=<idx>`**: 1-based bit index at which repeating begins (default `1`).

#### Flux-Quantum Pulse-Train (`SFQ`) — *[New in v2.8 Fork]*

**sfq(**[$T_D$&emsp;[*PER*&emsp;[*FWHM*&emsp;[*PATTERN_OR_COUNT*]]]]&emsp;[times=<$t_1,t_2,\dots$>]&emsp;[amp=<**VAL**>]&emsp;[phi0=<**MULT**>]&emsp;[r=<**REPEAT**>]**)**

Generates analytical $\Phi_0$-quantized Single-Flux-Quantum pulses without manually constructing `pwl(...)` coordinate tables:
- **On a Voltage Source (`V...`):** Emits raised-cosine ($\sin^2$) voltage pulses of duration $2w$ ($w = \text{FWHM}$, default $2\text{ ps}$) with exact area $\int_0^{2w} V(\tau)\,d\tau = \text{MULT} \cdot \Phi_0$ ($V_{\text{peak}} = \text{MULT} \cdot \Phi_0 / w \approx 1.0339\text{ mV}$ for $w = 2\text{ ps}$).
- **On a Phase Source (`P...`):** Emits $C^2$-smooth $2\pi \cdot \text{MULT}$ phase steps over duration $2w$.
- **On a Current Source (`I...`):** Emits raised-cosine current pulses of peak amplitude `amp=<val>` (default $500\,\mu\text{A}$).


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

**.subckt**&emsp;*SubcktName*&emsp;*IO Nodes*&emsp;[params:]&emsp;[*param1=default1* *param2=default2* ...]

...

**.ends**

The *SubcktName* specifies the name of the subcircuit and *IO Nodes* specify the which nodes within the subcircuit connects to outside.

A subcircuit can be used in the main netlist or another subcircuit (nesting) using the following syntax

**X**Label&emsp;*SubcktName*&emsp;*IO Nodes*&emsp;[params:]&emsp;[*param1=val1* ...]&emsp;(JSIM mode)

**X**Label&emsp;*IO Nodes*&emsp;*SubcktName*&emsp;[params:]&emsp;[*param1=val1* ...]&emsp;(WRspice (normal SPICE) mode)

Additionally, keywords in the form of **LABEL=VALUE** or **PARAM=VALUE** (including `{expression}` overrides) can be appended to the `.subckt` header (as default parameter values) or to the `X` instance line. Each `X` instance evaluates its own isolated parameter scope (`SubcktName|XLabel`), automatically recomputing all dependent `.param` expressions inside the subcircuit for that instance.

### Noise

As mentioned in the technical discussion, noise can be automatically inserted as current sources in parallel to each resistor. This thermal noise temperature and bandwidth can be specified globally using the following commands:

**.temp**&emsp;*Temperature in Kelvin*

**.neb**&emsp;*Bandwidth in Hertz*

### Spread & Batched Monte Carlo Robustness Analysis

JoSIM allows each value of inductors, resistors, capacitors and JJ (area/Ic) to be spread uniformly within a specified percentage range from the nominal value. Each time the value is used (in matrix creation), a new random value from the uniform distribution is chosen.

This allows for process variation and robustness margins to be simulated.

To set the spread globally, the following control needs to be set:

**.spread**&emsp;*Normalized percentage*&emsp;*[L=Inductor spread]*&emsp;*[B=JJ spread]*&emsp;*[C=Capacitor spread]*&emsp;*[R=Resistor spread]*&emsp;*[runs=N]*

Each individual component can also be spread by adding the **spread=** named parameter to the component declaration line.

The order of precedence is taken as local, specific global and then global. This means that if a global spread exists but  a global inductor specific spread also exists and the inductor being added has a local spread then the local spread will take precedence.

An example:

If an inductor has a value of *2pH*, this is its nominal value. If a local spread of *0.2* is specified this means a random value can be chosen anywhere between *1.6pH* and *2.4pH*. If global inductor specific spread is specified as *0.3* and the local spread is not defined, the value can be anything within the range *1.4pH -  2.6pH*. Similarly if the global spread is defined as *0.5* then the value each time the simulation is run can be anywhere between *1pH* and *3pH*.

#### Batched Monte Carlo Execution (`-b <N>`, `.option mc=<N>`, `.spread ... runs=<N>`) — *[New in v2.8 Fork]*

When `N > 1` Monte Carlo trials are requested via `-b <N>` (`--batch=<N>`), `.option mc=<N>` (`.option batch=<N>`), or `.spread ... runs=<N>`, JoSIM parses the netlist and performs symbolic graph partitioning **only once**, then executes all `N` independent fabrication-yield trials simultaneously in a single invocation using either:
- **Multi-Core OpenMP (`-p` / `--parallel`)**: Distributes independent Monte Carlo trials and circuit partitions across CPU threads using deterministic per-trial seeds (`baseSeed ^ (b * 0x9E3779B97F4A7C15ULL)`).
- **NVIDIA CUDA GPU (`-g` / `--gpu`, `.option gpu`)**: Packs all `N` trials into a coalesced batch-minor layout (`[baseIdx * N + b]`) and solves them in lockstep using the hardware-adaptive `advance_mc_coalesced_kernel` with shared symbolic CSR indices in on-chip shared memory.

By default, batched Monte Carlo outputs 4 ensemble statistics traces per requested `.print`/`.save` variable (computed via Welford's numerically stable online algorithm):
- `<Trace>`: Ensemble mean across all `N` trials
- `<Trace>_STD`: Sample standard deviation across all `N` trials
- `<Trace>_MIN`: Minimum envelope across all `N` trials
- `<Trace>_MAX`: Maximum envelope across all `N` trials

If `.option mcall` is specified, individual per-trial waveforms `<Trace>_MC0`, `<Trace>_MC1`, ..., `<Trace>_MC{N-1}` are also appended to the output file. In addition, a **Monte Carlo Robustness Summary** table is printed to standard output at the end of the run.

### Deterministic Parameter & Margin Sweeps (`.sweep`) — *[New in v2.8 Fork]*

**.sweep**&emsp;*Target[,Target2,...]*&emsp;[*LIN | MARGIN | LIST*]&emsp;*Args...*

Executes a 1D (or 2D Cartesian when two `.sweep` cards are given) deterministic parameter or margin sweep in a single batched invocation (`-p -n` or `-g`):
- **Targets:** `.param` names, category selectors (`@BIAS`, `@JJ`/`@B`, `@IND`/`@L`, `@RES`/`@R`, `@CAP`/`@C`, `@SRC`, `@ISRC`, `@VSRC`, `@PSRC`, `@GLOBAL`), device wildcards (`IB*`, `B*|X0`), or comma-separated multi-target groups (`Iebus,Vrail`).
- **Modes:**
  - `.sweep <target> <start> <stop> <step>`
  - `.sweep <target> LIN <count> <start> <stop>`
  - `.sweep <target> MARGIN <delta> [<count>]` (sweeps $[1-\delta, 1+\delta]$ around nominal)
  - `.sweep <target> LIST <v1> <v2> ... <vB>`

### Critical Operating Margin Finder & Event Verifier (`.margins` / `.verify_events`) — *[New in v2.8 Fork]*

**.margins**&emsp;[*Targets...*]&emsp;[range=<**DELTA**> | min=<**LO**> max=<**HI**>]&emsp;[n=<**NPTS**> | step=<**STEP**>]&emsp;[tol=<**TIME_TOL**>]&emsp;[file=<**CSV_PATH**>]

**.verify_events**&emsp;*GoldenEventsPath*&emsp;[tol=<**TIME_TOL**>]

Packs a shared nominal trial (`Trial 0`) and all target perturbations (`@DEFAULT` = `@BIAS,@JJ,@IND,@RES`, or `@DEVICES` / `@DEVICES|X0` for per-device ranking) into a single batched pass, verifying the exact ordered `(polarity, time)` SFQ event sequence within $\pm\text{tol}$ (default `50p`) and reporting ranked operating margins and root-cause failure modes.

### Digital Vector & SFQ Event Assertions (`.assert` / `.vec`) — *[New in v2.8 Fork]*

**.assert**&emsp;*TraceOrGlob*&emsp;**bits=***Pattern*&emsp;**td=***StartDelay*&emsp;**per=***BitPeriod*&emsp;[win=<**WINDOW**>]&emsp;[tol=<**TOL**>]

**.assert**&emsp;*TraceOrGlob*&emsp;**count=***ExpectedCount*&emsp;[from=<**TSTART**>]&emsp;[to=<**TSTOP**>]

**.assert**&emsp;*TraceOrGlob*&emsp;**times=***t1,t2,...*&emsp;[tol=<**TOL**>]

**.assert**&emsp;*TraceOrGlob*&emsp;**after**&emsp;*RefTrace*&emsp;[delay=<**DT**>]&emsp;[tol=<**TOL**>]

Verifies simulated SFQ switching events against digital specifications; exits with a non-zero status code if any assertion fails.

### SPICE Measurements with Native SFQ Triggers (`.meas` / `.measure`) — *[New in v2.8 Fork]*

**.meas tran**&emsp;*MeasName*&emsp;**trig** *TrigTrace* [sfq=<*k|last*> | val=<*v*> [rise=<*k*>|fall=<*k*>|cross=<*k*>]] **targ** *TargTrace* [sfq=<*k|last*> | val=<*v*> ...]

**.meas tran**&emsp;*MeasName*&emsp;**when** *Trace* [sfq=<*k|last*> | val=<*v*>]

**.meas tran**&emsp;*MeasName*&emsp;**find** *Trace* **at=***Time* | **when** *CondTrace* ...

**.meas tran**&emsp;*MeasName*&emsp;**avg | rms | min | max | min_at | max_at | pp | integ | count | freq | vpeak | fwhm | dt_pi | area** *Trace* [from=<*t1*>] [to=<*t2*>] [sfq=<*k*>]

**.meas tran**&emsp;*MeasName*&emsp;**param=**'*Expression*'&emsp;[val=<*target*> tol=<*tol*> | min=<*lo*> max=<*hi*>]

Evaluated measurement values are stored into `.param` scope (accessible to subsequent `.meas` expressions and `pyjosim`) and printed at the end of the run (or exported via `--meas-file=<csv>`).

### Initial Conditions (`.ic`) & Topological $\beta_L$ Loop Auditor (`.check_loops`) — *[New in v2.8 Fork]*

**.ic**&emsp;[P(*node*)=*rad*]&emsp;[V(*node*)=*volts*]&emsp;[I(*Lname*)=*amps*] ...

**.check_loops**&emsp;[strict]&emsp;[file=<*path.csv|.json*>]

`.check_loops` (or `--check-loops`) discovers all fundamental superconducting `B + L` loops, computes $L_{\text{loop}}$, $\beta_L = 2\pi L_{\text{loop}} I_{c,\min}/\Phi_0$, and net static flux offset $|\Phi_{\text{offset}}|/\Phi_0$, and exports `LOOP_COUNT`, `LOOP_STORAGE_COUNT`, `LOOP_DEGENERATE_COUNT`, `BETA_L_<JJ>`, `L_LOOP_<JJ>`, and `FLUX_LOOP_<JJ>` into `.param` scope.

### Timing Jitter, Skew & Bit-Error-Rate Analyzer (`.jitter` / `.ber`) — *[New in v2.8 Fork]*

**.jitter**&emsp;[*Name*]&emsp;*TargetTraceOrGlob*&emsp;[ref=<*RefTrace*> | per=<*ClockPeriod*> [td=<*Start*>] [bits=<*Pattern*>]]&emsp;[max_rms=<*t*>]&emsp;[max_pp=<*t*>]&emsp;[max_ber=<*b*>]&emsp;[strict]

Computes sub-timestep SFQ propagation latency, RMS jitter ($\sigma_t$), peak-to-peak jitter, cycle-to-cycle jitter, and bit-error rate (BER), exporting `LATENCY_<Name>`, `JITTER_RMS_<Name>`, `JITTER_PP_<Name>`, `JITTER_C2C_<Name>`, `BER_<Name>`, and `ERRORS_<Name>` into `.param` scope.

### Batched OAAT Sensitivity & Automated Margin Centering (`.sens` / `.sensitivity`) — *[New in v2.8 Fork]*

**.sens**&emsp;[*Targets...*]&emsp;[delta=<*frac*>]&emsp;[metric=<*MeasName*>]&emsp;[center [min=<*lo*>] [max=<*hi*>] [step=<*st*>]]&emsp;[goal=<*target_val*>]&emsp;[file=<*path.csv|.json*>]

Runs a single batched pass of $2D + 1$ central-difference perturbations (plus optional full operating-margin brackets when `CENTER` is specified) to rank parameter sensitivities, RSS variance weights $W_k$, and recommended margin-centering updates.

### Options and Reproducibility (`.option`)

Simulator options can be configured in the netlist using the `.option` (or `.options`) control card:

**.option**&emsp;*OPTION_KEY*[=*VALUE*]&emsp;[*OPTION_KEY2*[=*VALUE2*] ...]

| Syntax | Origin | Description |
| :--- | :---: | :--- |
| `.option seed=<int>` | Inherited (`v2.7`) / Enhanced (`v2.8`) | Sets a deterministic RNG seed for **noise** and **spread** (also settable via `JOSIM_SEED` env var). |
| `.option nodal[=1\|2\|auto]` | **New (`v2.8`)** | Enables the **Schur-Complement Nodal Admittance Solver** (`-n`): `auto` (fill-aware), `1` ($R+L$/$R+R$), `2` ($+L+L$). |
| `.option events[=file]` | **New (`v2.8`)** | Extracts $2\pi$ SFQ switching events (`-e [file]`; use `.option eventsonly[=file]` for zero-allocation Events-Only mode). |
| `.option eventmetrics[=spec]` | **New (`v2.8`)** | Selects SFQ pulse metrics (`-E [spec]`, alias: `.option eventpulse`): `pulse`, `all`, `t_peak`, `v_peak`, `fwhm`, `dt_pi`, `area`, `time=peak`. |
| `.option lumpptl[=0\|1]` | **New (`v2.8`)** | Replaces sub-timestep lossless PTLs ($T_d < 0.5\Delta t$) with matched $L$-$C$ $\pi$-equivalents (disable: `.option nolumpptl`). |
| `.option minptl=<val>` | **New (`v2.8`)** | Custom delay threshold $T_{d,\min}$ for `.option lumpptl`. |
| `.option rollback[=0\|1\|2\|window\|stay]` | **New (`v2.8`)** | In-memory rolling checkpoint rollback on $>0.4\pi$ phase jumps (`1`/`window` default, `2`/`stay`, `0`/`.option norollback`). |
| `.option gpu[=1\|2\|3\|4]` | **New (`v2.8`)** | Enables CUDA GPU solver (`-g [1\|2\|3\|4]`): `1` = Auto, `2` = Hybrid, `3` (`gpumono`/`cusolverrf`) = Tier 3 Monolithic MNA, `4` (`gpunodal`) = Tier 4 Monolithic Nodal. |
| `.option bypass` | **New (`v2.8`)** | Skips `klu_l_tsolve` on quiescent PTL partitions during OpenMP windowed simulation (`-p`). Alias: `.option latency`. |
| `.option bypasstol=<val>` | **New (`v2.8`)** | Activity threshold for `.option bypass` (default `1e-12`). |
| `.option mc=<N>` | **New (`v2.8`)** | Executes `N` batched Monte Carlo (`.spread`) trials in one run (`-b <N>`, aliases: `batch`, `runs`). |
| `.option mcall` | **New (`v2.8`)** | Appends individual trial waveforms (`<Trace>_MC0 .. MC{N-1}`) to batched Monte Carlo or `.sweep` output (alias: `sweepall`). |
| `.option sweep=<spec>` | **New (`v2.8`)** | Configures a deterministic parameter/margin sweep (`-S <spec>`). |
| `.option margins[=<spec>]` | **New (`v2.8`)** | Enables critical operating margin analysis (`-M`; also `marginfile=<f>`, `margintol=<t>`, `verifyevents=<f>`). |
| `.option assert=<spec>` | **New (`v2.8`)** | Adds a digital SFQ event assertion (`-A <spec>`). |
| `.option meas=<spec>` | **New (`v2.8`)** | Adds a `.meas` expression (`--meas=<spec>`, `measfile=<f>`). |
| `.option ic=<spec>` | **New (`v2.8`)** | Adds an initial condition (`--ic=<spec>`). |
| `.option checkloops[=strict]` | **New (`v2.8`)** | Enables the topological $\beta_L$ loop auditor (`--check-loops`, `loopfile=<f>`). |
| `.option jitter[=<spec>]` | **New (`v2.8`)** | Enables SFQ timing jitter & BER analysis (`-J`, `jitterfile=<f>`). |
| `.option sens[=<spec>]` | **New (`v2.8`)** | Enables batched OAAT sensitivity & margin centering (`--sens`, `sensfile=<f>`). |
| `.option txinterp` | **New (`v2.8`)** | Linearly interpolates fractional transmission-line delays (or per-line `interp=1`; alias: `fractx`). |
| `.option kcutoff=<val>` | **New (`v2.8`)** | Ignores mutual inductance entries (`K`) with $\|k\| < \text{val}$ during matrix setup (alias: `kmin`). |
| `.option corrector=<N>` | **New (`v2.8`)** | Enables up to `N` implicit predictor-corrector JJ iterations per step (`-c <N>`). |
| `.option correctortol=<v>` | **New (`v2.8`)** | Phase convergence threshold (radians, default `1e-3`) for `corrector`. |
| `.option fixedstep` | **New (`v2.8`)** | Disables automatic timestep halving on $>0.4\pi$ phase jumps (`-f`). |
| `.option warnstep` | **New (`v2.8`)** | Prints diagnostic warning naming the JJ and time that triggered step halving. |

Examples:
```cir
.option seed=4190754512324517577
.option nodal
.option txinterp warnstep
.option corrector=2 correctortol=1e-4
.option kcutoff=1e-3
.option gpu=4
```

### State Checkpointing (`.savestate` / `.loadstate`) — *[New in v2.8 Fork]*

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

### IV Curve, RF Shapiro Steps & 2D SQUID Flux Modulation (`.iv` & `.squid`)

JoSIM allows the user to output an IV curve for a specified JJ model or whole-circuit bias source within the netlist using the following commands:

**.iv**&emsp;*modelname | isrc_name | CIRCUIT*&emsp;*max_current*&emsp;[*filepath*]&emsp;[min=<*IMIN*>]&emsp;[step=<*ISTEP*>]&emsp;[rshunt=<*R*>]&emsp;[rf_amp=<*IRF*> rf_freq=<*FRF*>]&emsp;[vnode=<*NODE_OR_PAIR*>]&emsp;[name=<*TAG*>]&emsp;[out=<*FILEPATH*>]

**.squid**&emsp;*ib_source*&emsp;*loop_inductor | flux_source*&emsp;[imax=<*IMAX*>]&emsp;[imin=<*IMIN*>]&emsp;[istep=<*ISTEP*>]&emsp;[fmin=<*FMIN*>]&emsp;[fmax=<*FMAX*>]&emsp;[fstep=<*FSTEP*>]&emsp;[vnode=<*NODE_OR_PAIR*>]&emsp;[name=<*TAG*>]&emsp;[out=<*FILEPATH*>]

- **Model / Whole-Circuit / Shapiro-Step `.iv`:** Sweeps bias current (in parallel across OpenMP threads) and extracts $I_{c+}$, $I_{c-}$, retrapping currents $I_{r\pm}$, normal resistance $R_n$, and (when `rf_amp=` and `rf_freq=` are provided) quantized microwave Shapiro step widths $\Delta I_0, \Delta I_1, \Delta I_2, \Delta I_3$ at $V_n = n \Phi_0 f_{\text{RF}}$.
- **1D & 2D SQUID Flux Modulation (`.squid`):** Sweeps external magnetic flux $\Phi_{\text{ext}} \in [\text{FMIN}, \text{FMAX}]$ (in $\Phi_0$ on loop inductor `L` or current on `flux_source`) at fixed bias (`ib=<val>`) or across a 2D current grid (`imin=`, `imax=`, `istep=`), automatically exporting `SQUID_VMIN`, `SQUID_VMAX`, `SQUID_DV`, `SQUID_DV_DPHI_MAX`, `SQUID_PHI_OPT`, `SQUID_IB_OPT`, `SQUID_IC_MAX`, `SQUID_IC_MIN`, and `SQUID_IC_MOD` into `.param` scope.

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

Wildcard globs (`*` matching zero or more characters, `?` matching a single character) are also supported in `.print`, `.plot`, and `.save` (*[New in v2.8 Fork]*), e.g.:

**.print** P(B*|X0) V(*|X0) I(IB?|X*)

This method follows the path from inside to out when nested subcircuits are used. 

Subcircuits as mentioned before can also be nested almost indefinitely as they are expanded upon simulation.

If the label is not immediately apparent and required for output, the simulation can be run using the `-V 3` cli option to show the expanded main design, allowing the exact label name to be identified. 

### File

Multi-file output can be achieved by separating output commands with the following command:

**.file**&emsp;*filepath*

This indicates that every output request that follows this command needs to be stored in the file specified by *filepath*. Both `.file` and the `-o` (`--output=`) command-line switch automatically select the output format from the file extension:

| Extension | Format | Origin | Description |
| --------- | ------ | :---: | ----------- |
| `.csv` | Comma-Separated Values (ASCII) | Inherited (`v2.7`) | Standard comma-separated text table with header row (`time,...`). |
| `.dat` | Space-Separated Values (ASCII) | Inherited (`v2.7`) | Standard space-separated text table with header row. |
| `.raw` or no extension | SPICE3 Raw (ASCII) | Inherited (`v2.7`) | Standard ASCII SPICE rawfile (`Values:` section). |
| `.bin` | Native JoSIM Binary (`JOSIMBIN`) | **New in Fork (`v2.8`)** | High-speed column-major IEEE-754 `float64` binary format with self-describing trace metadata header. ~37x faster to write and ~38% smaller than `.csv`. |
| `.npy` | NumPy Structured Array (`.npy`) | **New in Fork (`v2.8`)** | Standard NumPy v1.0/v2.0 structured array with named `<f8` columns, directly loadable via `numpy.load()` or `scripts/josim_io.py`. |
| `.rawb` or `.braw` | SPICE3 Binary Raw (`Binary:`) | **New in Fork (`v2.8`)** | Standard SPICE3 binary rawfile with ASCII preamble followed by interleaved IEEE-754 `float64` records. |

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

**.param** &emsp;*VarName*=*Expression* [*VarName2*=*Expression2* ...]

*VarName* is the variable name that can be used anywhere else in the circuit and *Expression* is a mathematical expression that is evaluated using an implementation of Dijkstra's shunting yard algorithm, whereby the expression is converted into reverse polish notation (RPN) and evaluated.

Additionally, expressions can also contain other variables and parameters will be continuously evaluated until all variables are reduced to values. If variables are not defined the program will halt and produce an error.

In **`v2.8`**, mathematical expressions enclosed in curly braces (`{2*IBIAS + 10u}`) can also be used inline on component cards, source waveforms (`PWL`, `PULSE`, `SIN`, `PAT`, `SFQ`), and subcircuit instance overrides. Supported built-in functions include:
- **1-argument functions:** `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `exp`, `exp2`, `ln`/`log`, `log10`, `log2`, `sqrt`, `cbrt`, `abs`/`fabs`, `sgn`/`sign`, `floor`, `ceil`, `round`/`nint`, `trunc`/`int`, `frac`, `step`/`u`, `uramp`, `erf`, `erfc`, `db`.
- **2-argument & variadic functions:** `min(a, b, ...)`, `max(a, b, ...)`, `pow(x, y)` / `pwr(x, y)` / `pwrs(x, y)`, `atan2(y, x)`, `hypot(x, y)`, `mod(x, y)` / `fmod(x, y)`.
- **3-argument functions:** `if(cond, tVal, fVal)` / `ternary(cond, tVal, fVal)`, `limit(x, lo, hi)` / `clamp(x, lo, hi)`.

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
