#!/usr/bin/gnuplot

### --- Settings ---
set terminal qt 1 size 1060,600    # qt gives separate interactive windows
set grid
set key top right
set style line 1 lw 2.0 lc 1
set style line 2 lw 1.5 lc 2

# File names

errHP = "fcc_ee_test_simulation/tbt_simu/sim_tbt/HmaxAPJphase_nofilt.sdds"
corrHP = "fcc_ee_corrections/tbt_simu/sim_tbt/HmaxAPJphase_nofilt.sdds"

# Filter: keep only rows where col1 == 1
filtV(x,val) = (x==1 ? val : 1/0)
filtH(x,val) = (x==0 ? val : 1/0)

# ### Plot 2

set terminal qt 1
set title "H Phase"
set xlabel "s"
set ylabel "P"

plot \
    errHP u (filtH($1,$3)):(filtH($1,$4)) w l ls 1 title "errors", \
    corrHP u (filtH($1,$3)):(filtH($1,$4)) w l ls 2 title "errors+corrections"

pause -1
