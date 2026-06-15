module parameter
  implicit none
  integer, parameter :: lx = 2, ly = 1, lxy = lx * ly !gedian
  integer, parameter :: NUP = 1, NDN = 1, NELEC = NUP + NDN, NSPIN = 2
  integer :: trial_read_mode = 0
  integer :: geom_mode = 0
  real(sp) :: h_pin = 0.0
  real(sp) :: theta_x_val = 0.0
  real(sp) :: theta_y_val = 0.0
  real(sp) :: pin_right_factor = 1.0
  integer :: tabc_mode = 0
  integer :: pair_source_mode = 0
  integer :: pair_frame_mode = 0
end module parameter
