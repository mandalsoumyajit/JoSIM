* ngspice + JoSIM XSPICE Co-Simulation Example 2:
* JoSIM SFQ-Triggered Electrothermal nTron Nanowire Driving a 2-Stage 4K Cryo-CMOS LNA
* Run with:
*   LD_LIBRARY_PATH=/tmp/josim-build ngspice -b cosim/ngspice/examples/ntron_to_cmos_lna.sp

.model nmos_4k nmos(level=1 vto=0.20 kp=650u gamma=0.08 phi=0.6 lambda=0.02 cbd=0.2f cbs=0.2f cgso=20p cgdo=20p)

* 1-Port JoSIM nTron Subsystem (Channel node "nd" exposed to ngspice)
A_ntron [%gd(nd 0)] josim_ntron_mod
.model josim_ntron_mod josim_cosim(
+ netlist="cosim/spectre/examples/ntron_driver.cir"
+ pos_nodes=["nd"]
+ neg_nodes=["0"]
+ port_modes=[0]
+ phase_mode=1
+ out_csv="ntron_driver_josim.csv"
+ )

* 2-Stage 4K Cryo-CMOS Common-Source + Source-Follower LNA
Vdd    vdd    0 1.0
Vbias1 vbias1 0 0.28
Cc1    nd     g1 2.0p
Rbias1 vbias1 g1 4k
M_cs   d1     g1 0 0 nmos_4k w=24u l=0.12u
Rd1    vdd    d1 1.2k
Cload1 d1     0  0.5f

Vbias2 vbias2 0 0.55
Cc2    d1     g2 2.0p
Rbias2 vbias2 g2 6k
M_sf   vdd    g2 vout 0 nmos_4k w=32u l=0.12u
Rsf    vout   0  800
Cload2 vout   0  1.0f

.control
pre_codemodel /tmp/josim-build/josim_cosim.cm
set wr_singlescale
tran 0.2p 60p
wrdata ntron_to_cmos_lna_ng.dat v(nd) v(d1) v(vout)
quit
.endc
.end
