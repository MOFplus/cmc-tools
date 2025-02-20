module ric_stretches

    use ric_common
    use ric_math

    implicit none

contains

    subroutine ric_stretches_bmat(hmat,coords,defs,ibrs,bmat,vals)
        real(dp), intent(in)    :: hmat(3,3)
        real(dp), intent(in)    :: coords(:,:)
        integer , intent(in)    :: defs(:,:)
        integer , intent(in)    :: ibrs(:)
        real(dp), intent(inout) :: bmat(:,:)
        real(dp), intent(inout) :: vals(:)

        integer  :: nstretch, i
        !f2py intent(in) hmat, coords, defs, ibrs
        !f2py intent(inout) bmat, vals

        ! Get the number of stretches
        nstretch = size(defs,dim=2)

        do i = 1, nstretch
            call ric_strech(hmat,coords,defs(:,i),bmat(:,ibrs(i)),vals(i))
        end do

    end subroutine 

    ! This is from ff_gen_internals.f (strech)
    subroutine ric_strech(hmat,coords,def,bmat,val)

        real(dp), intent(in)    :: hmat(3,3)
        real(dp), intent(in)    :: coords(:,:)
        integer , intent(in)    :: def(2)
        real(dp), intent(inout) :: bmat(:)
        real(dp), intent(out)   :: val

        integer  :: ia1(3), ia2(3), i
        real(dp) :: a1(3), a2(3), v(3)

        a1 = coords(:,def(1))
        a2 = coords(:,def(2))

        call rel_vec(hmat,a1,a2,norm_vec=v,dist=val)

        ia1 = (/( 3*(def(1)-1)+i, i=1,3 )/)
        ia2 = (/( 3*(def(2)-1)+i, i=1,3 )/)

        bmat(ia1) = -v
        bmat(ia2) =  v

    end subroutine

end module
