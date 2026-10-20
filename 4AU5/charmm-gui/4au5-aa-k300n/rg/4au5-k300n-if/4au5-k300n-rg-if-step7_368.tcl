
for {set i 0 } {$i < [molinfo top get numframes]} {incr i} {

   set sel [atomselect top "resid 134 167 168 345 72 and sidechain and noh" frame $i]
 
 # make sure this is a proper selection and has atoms
  if {[$sel num] <= 0} {
    error "gyr_radius: must have at least one atom in selection"
  }
  
  set center_of_mass [measure center $sel weight mass]

  # gyration is sqrt( sum((r(i) - r(center_of_mass))^2) / N)
  set com $center_of_mass
  set sum 0
  foreach coord [$sel get {x y z}] {
    set sum [vecadd $sum [veclength2 [vecsub $coord $com]]]
  }
  set rg [ expr sqrt($sum / ([$sel num] + 0.0))]

  exec echo $rg >> 4au5-k300n-rg-if-step7_368.dat
}
