#!/usr/bin/env fish

set scripts plotVP.gp plotVJ.gp plotHP.gp plotHJ.gp

for script in $scripts
    gnuplot tools/$script &
end

wait
