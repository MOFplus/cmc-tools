module ric_math

  use ric_common

  implicit none

  public :: rel_vec
  public :: cdc
  public :: cross_prod
  public :: norm
  public :: normalize

contains

  subroutine rel_vec(hmat,a1,a2,vec,norm_vec,dist)
    real(dp),           intent(in)  :: hmat(3,3), a1(3), a2(3)
    real(dp), optional, intent(out) :: vec(3), norm_vec(3), dist

    real(dp):: vec_(3), norm_vec_(3), dist_

    if (all(hmat == 0.0_dp)) then
      call relvec_0d(vec_,norm_vec_,a1,a2,dist_)
    else
      call relvec_3d(hmat,vec_,norm_vec_,a1,a2,dist_)
    end if

    if (present(vec     )) vec      = vec_
    if (present(norm_vec)) norm_vec = norm_vec_
    if (present(dist    )) dist     = dist_

  end subroutine

!     From ff_gen_math.f
!     Takes two vectors c1 and c2

!     Return ee vector pointing from c1 to c2
!     Return e normal vector of ee
!     Return r length of ee

  subroutine relvec_0d(ee,e,c1,c2,r)
    real(dp), intent(in)  :: c1(3), c2(3)
    real(dp), intent(out) :: ee(3), e(3), r

    ee = c2 - c1
    r  = sum(ee*ee)
    if(r <= 1.e-12_dp) return
    r=sqrt(r)
    e=ee/r

  end subroutine

!     From ff_gen_math.f
!     Takes two vectors c1 and c2
!     Takes lvec lattice vectors as defined in boxes.i

!     Return ee vector pointing from c1 to c2
!     Return e normal vector of ee
!     Return r length of ee

!     PBC is considered by looking for the shortest ee vector

  subroutine relvec_3d(lvec,ee,e,c1,c2,r)
    real(dp), intent(in)  :: lvec(3,3), c1(3), c2(3)
    real(dp), intent(out) :: ee(3), e(3), r

    real(dp) :: c2_imig(3), ee_tmp(3), r_tmp
    integer  :: x, y, z

    r = huge(r)
    ee = 0._dp

    do x = -1,1
      do y = -1,1
        do z = -1,1
          c2_imig = c2 + x*lvec(1,:) + y*lvec(2,:) + z*lvec(3,:)
          ee_tmp = c2_imig - c1
          r_tmp = sum(ee_tmp*ee_tmp)
          if (r_tmp < r) then
            r = r_tmp
            ee = ee_tmp
          end if
        end do
      end do
    end do

    r=sqrt(r)
    e = ee/r

  end subroutine

!     From ff_gen_math.f
!       Take three vectors a1, a2, and a3,
!       which define an angle a1-a2-a3

!       Return the dot product of two normal vectors defining angle.

  function cdc(hmat,a1,a2,a3)
    real(dp), intent(in) :: hmat(3,3), a1(3), a2(3), a3(3)
    real(dp)             :: cdc

    real(dp) :: v1(3), v2(3)

    call rel_vec(hmat,a2,a1,norm_vec=v1)
    call rel_vec(hmat,a2,a3,norm_vec=v2)

    cdc = dot_product(v1,v2)
    cdc = min(max(cdc,-1._dp),1._dp)

  end function

  pure function cross_prod(a,b)
    real(dp), intent(in) :: a(3), b(3)
    real(dp)             :: cross_prod(3)

    cross_prod = (/ a(2)*b(3) - a(3)*b(2), &
                    a(3)*b(1) - a(1)*b(3), &
                    a(1)*b(2) - a(2)*b(1) /)

  end function

  pure function norm(a)
    real(dp), intent(in) :: a(3)
    real(dp)             :: norm

    norm = sqrt(dot_product(a,a))

  end function

  pure function normalize(a)
    real(dp), intent(in):: a(3)
    real(dp)            :: normalize(3)

    normalize = a/sqrt(sum(a**2))

  end function

end module

