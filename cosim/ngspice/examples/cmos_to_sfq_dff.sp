* ngspice + JoSIM XSPICE Co-Simulation Example 1:
* 4K Cryo-CMOS DATA & CLOCK MOSFET Drivers -> 2-Port JoSIM RSFQ D-Flip-Flop (DRO)
* Run with:
*   LD_LIBRARY_PATH=/tmp/josim-build ngspice -b cosim/ngspice/examples/cmos_to_sfq_dff.sp

.model nmos_4k nmos(level=1 vto=0.18 kp=600u gamma=0.1 phi=0.6 lambda=0.02 cbd=0.5f cbs=0.5f cgso=40p cgdo=40p)
.model pmos_4k pmos(level=1 vto=-0.18 kp=300u gamma=0.1 phi=0.6 lambda=0.02 cbd=0.5f cbs=0.5f cgso=40p cgdo=40p)

Vdd vdd 0 0.8

* Active-low DATA input pulse (at t = 18..22 ps) and CLOCK readout pulse (at t = 44..48 ps)
Vdin din_in 0 pwl(0 0.8 17p 0.8 18p 0.0 22p 0.0 23p 0.8 75p 0.8)
Vclk clk_in 0 pwl(0 0.8 43p 0.8 44p 0.0 48p 0.0 49p 0.8 75p 0.8)

* Cryo-CMOS Inverter + Source-Follower Drivers + Bias/Attenuation Network
Mp_d   drv_d din_in vdd vdd pmos_4k w=12u l=0.12u
Mn_d   drv_d din_in 0   0   nmos_4k w=6u  l=0.12u
Msf_d  vdd   drv_d  sf_d 0  nmos_4k w=18u l=0.12u
Ratt_d sf_d  sfq_set 1800

Mp_c   drv_c clk_in vdd vdd pmos_4k w=12u l=0.12u
Mn_c   drv_c clk_in 0   0   nmos_4k w=6u  l=0.12u
Msf_c  vdd   drv_c  sf_c 0  nmos_4k w=18u l=0.12u
Ratt_c sf_c  sfq_ro  1800

* 2-Port JoSIM RSFQ DFF Subsystem (%gd = voltage-controlled current port array)
A_dff [%gd(sfq_set 0) %gd(sfq_ro 0)] josim_dff_mod
.model josim_dff_mod josim_cosim(
+ netlist="cosim/spectre/examples/sfq_dff.cir"
+ pos_nodes=["n_set" "n_ro"]
+ neg_nodes=["0" "0"]
+ port_modes=[0 0]
+ phase_mode=1
+ out_csv="sfq_dff_josim.csv"
+ )

.control
pre_codemodel /tmp/josim-build/josim_cosim.cm
set wr_singlescale
tran 0.25p 75p
wrdata cmos_to_sfq_dff_ng.dat v(drv_d) v(drv_c) v(sfq_set) v(sfq_ro)
quit
.endc
.end
