for {set i 0} {$i < [molinfo top get numframes]} {incr i} {

    set core  [atomselect top "(resid 95 to 116 or resid 121 to 131 or resid 134 to 143 or resid 150 to 175 or resid 290 to 311 or resid 327 to 336 or resid 340 to 350 or resid 357 to 382) and protein and noh" frame $i]
    set dimer [atomselect top "(resid 12 to 30  or resid 182 to 200 or resid 59  to 85  or resid 205 to 218 or resid 223 to 236 or resid 247 to 271) and protein and noh" frame $i]

    set center_core [measure center $core weight mass]
    set center_dimer [measure center $dimer weight mass]

    exec echo $center_core $center_dimer_step7_347 >> core_dimer_position_step7_347.dat

}
