#!/usr/bin/env gnuplot

temps = ARG1."/temps"
temps_avg = ARG1."/temps_avg"

# Set the output format and file name
set terminal svg enhanced size 800,1000 font "serif,14"
set out ARG1."/ising_energy_magnet.svg"

# Set overall plot title
set title "Ising Model Results"

# Define the layout for the subplots
set multiplot layout 2,1

# First subplot: Energy vs. Temperature
set title "Energy vs. Temperature"
set xlabel "Temperature"
set ylabel "Energy"
set grid
set key top left
plot temps using 1:3 title "Energy" lt rgb "gray", \
     temps_avg using 1:2 title "Mean value" with lines lt rgb "blue" lw 4

# Second subplot: Magnetization vs. Temperature
set title "Magnetization vs. Temperature"
set xlabel "Temperature"
set ylabel "Magnetization"
set grid
set key top right
plot temps using 1:4 title "Magnetization" lt rgb "gray", \
     temps_avg using 1:3 title "Mean value" with lines lt rgb "red" lw 4

# Restore the default layout
unset multiplot
