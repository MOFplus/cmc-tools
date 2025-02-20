module ric_in_bends

    use ric_common
    use ric_math

    implicit none

    contains


    subroutine ric_in_bends_bmat(hmat,coords,defs,ibrs,bmat,vals)
        real(dp), intent(in)    :: hmat(3,3)
        real(dp), intent(in)    :: coords(:,:)
        integer , intent(in)    :: defs(:,:)
        integer , intent(in)    :: ibrs(:)
        real(dp), intent(inout) :: bmat(:,:)
        real(dp), intent(inout) :: vals(:)

        !f2py intent(in) hmat, coords, defs, ibrs
        !f2py intent(inout) bmat, vals

        integer :: nin_bend, i

        ! Get the number of in-plane bends
        nin_bend = size(defs,dim=2)

        do i = 1, nin_bend
            call ric_in_bend(hmat,coords,defs(:,i),bmat(:,ibrs(i)),vals(i))
        end do


    end subroutine

    ! This is from ff_gen_internals.f (bendin)
    subroutine ric_in_bend(hmat,coords,def,bmat,val)
        real(dp), intent(in)    :: hmat(3,3)
        real(dp), intent(in)    :: coords(:,:)
        integer , intent(in)    :: def(3)
        real(dp), intent(inout) :: bmat(:)
        real(dp), intent(out)   :: val

        integer  :: ia1(3), ia2(3), ia3(3), i
        real(dp) :: a1(3), a2(3), a3(3), v1(3), v2(3), d1, d2, axis(3)

        ! Get atom positions
        a1 = coords(:,def(1))
        a2 = coords(:,def(2))
        a3 = coords(:,def(3))

        ! Compute angle vectors
        call rel_vec(hmat,a2,a1,norm_vec=v1,dist=d1)
        call rel_vec(hmat,a2,a3,norm_vec=v2,dist=d2)

        ! Compute angle values
        val = dot_product(v1,v2)
        val = min(max(val,-1._dp),1._dp)
        val = acos(val)

        ! Compute anlge axis
        axis = cross_prod(v1,v2) ! axis = v1 x v2
        axis = axis/sqrt(sum(axis**2))

        ! Compute B-matrix indices
        ia1 = (/( 3*(def(1)-1)+i, i=1,3 )/)
        ia2 = (/( 3*(def(2)-1)+i, i=1,3 )/)
        ia3 = (/( 3*(def(3)-1)+i, i=1,3 )/)

        ! Compute B-matrix elements
        bmat(ia1) =  cross_prod(v1,axis)/d1
        bmat(ia3) = -cross_prod(v2,axis)/d2
        bmat(ia2) = -(bmat(ia1) + bmat(ia3))

    end subroutine

end module
