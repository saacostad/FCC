#!/usr/bin/gnuplot

### --- Settings ---
set terminal qt 0 size 900,600    # qt gives separate interactive windows
set grid
set key top right
set style line 1 lw 2.0 lc 1
set style line 2 lw 1.5 lc 2

# File names
errVJ = "fcc_ee_test_simulation/tbt_simu/sim_tbt/VmaxAPJaction_nofilt.sdds"
corrVJ = "fcc_ee_corrections/tbt_simu/sim_tbt/VmaxAPJaction_nofilt.sdds"

errHJ = "fcc_ee_test_simulation/tbt_simu/sim_tbt/HmaxAPJaction_nofilt.sdds"
corrHJ = "fcc_ee_corrections/tbt_simu/sim_tbt/HmaxAPJaction_nofilt.sdds"

errHP = "fcc_ee_test_simulation/tbt_simu/sim_tbt/HmaxAPJphase_nofilt.sdds"
corrHP = "fcc_ee_corrections/tbt_simu/sim_tbt/HmaxAPJphase_nofilt.sdds"

errVP = "fcc_ee_test_simulation/tbt_simu/sim_tbt/VmaxAPJphase_nofilt.sdds"
corrVP = "fcc_ee_corrections/tbt_simu/sim_tbt/VmaxAPJphase_nofilt.sdds"

# Filter: keep only rows where col1 == 1
filtV(x,val) = (x==1 ? val : 1/0)
filtH(x,val) = (x==0 ? val : 1/0)

### --- Plot 1 ---
set terminal qt 1
set title "V Action"
set xlabel "s"
set ylabel "J"

plot \
    errVJ u (filtV($1,$3)):(filtV($1,$4)) w l ls 1 title "errors", \
    corrVJ u (filtV($1,$3)):(filtV($1,$4)) w l ls 2 title "errors+corrections"


### --- Plot 2 ---
set terminal qt 2
set title "H Action"
set xlabel "s"
set ylabel "J"

plot \
    errHJ u (filtH($1,$3)):(filtH($1,$4)) w l ls 1 title "errors", \
    corrHJ u (filtH($1,$3)):(filtH($1,$4)) w l ls 2 title "errors+corrections"


### --- Plot 3 ---
set terminal qt 3
set title "V Phase"
set xlabel "s"
set ylabel "P"

plot \
    errVP u (filtV($1,$3)):(filtV($1,$4)) w l ls 1 title "errors", \
    corrVP u (filtV($1,$3)):(filtV($1,$4)) w l ls 2 title "errors+corrections"


### --- Plot 4 ---
set terminal qt 4
set title "H Phase"
set xlabel "s"
set ylabel "P"

plot \
    errHP u (filtH($1,$3)):(filtH($1,$4)) w l ls 1 title "errors", \
    corrHP u (filtH($1,$3)):(filtH($1,$4)) w l ls 2 title "errors+corrections"

pause -1 "enter or whatever"
