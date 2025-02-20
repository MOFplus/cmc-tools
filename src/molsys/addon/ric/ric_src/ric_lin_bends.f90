module ric_lin_bends

    use ric_common
    use ric_math

    implicit none

contains 


    subroutine ric_lin_bends_bmat(hmat,coords,defs,inds,axes,ibrs,bmat,vals)
        implicit none
        real(dp), intent(in)    :: hmat(3,3)
        real(dp), intent(in)    :: coords(:,:)
        integer , intent(in)    :: defs(:,:)
        integer , intent(in)    :: inds(:)
        real(dp), intent(inout) :: axes(:,:)
        integer , intent(in)    :: ibrs(:)
        real(dp), intent(inout) :: bmat(:,:)
        real(dp), intent(inout) :: vals(:)

        integer  :: nlin_bend, i
        real(dp) :: a1(3), a2(3), a3(3), a4(3), vec(3), ref(3)

        !f2py intent(in) hmat, coords, defs, inds, aces, ibrs
        !f2py intent(inout) bmat, vals

        ! Get the number of linear bends
        nlin_bend = size(defs,dim=2)

        do i = 1, nlin_bend

            if (inds(i) /= 0) then
                a1 = coords(:,defs(1,i))
                a2 = coords(:,defs(2,i))
                a3 = coords(:,defs(3,i))
                a4 = coords(:,abs(inds(i)))

                call rel_vec(hmat,a1,a3,norm_vec=vec)
                call rel_vec(hmat,a2,a4,norm_vec=ref)

                if (inds(i) > 0) then
                    axes(:,i) = normalize(cross_prod(vec,cross_prod(ref,vec)))
                else
                    axes(:,i) = normalize(cross_prod(ref,vec))
                end if

            end if

            !write(*,*) 'axis', axes(:,i)

            call ric_lin_bend(hmat,coords,defs(:,i),axes(:,i),bmat(:,ibrs(i)),vals(i))

        end do


    end subroutine

    ! This is from ff_gen_internals.f (bend10)
    subroutine ric_lin_bend(hmat,coords,def,axis,bmat,val)

        use ric_common
        use ric_math

        implicit none
        real(dp), intent(in)    :: hmat(3,3)
        real(dp), intent(in)    :: coords(:,:)
        integer , intent(in)    :: def(3)
        real(dp), intent(in)    :: axis(3)
        real(dp), intent(inout) :: bmat(:)
        real(dp), intent(out)   :: val

        integer  :: ia1(3), ia2(3), ia3(3), i
        real(dp) :: a1(3), a2(3), a3(3), v1(3), v2(3), d1, d2, ax(3), &
            n1(3), n2(3), dn1, dn2, nn(3)

        ! Get atom positions
        a1 = coords(:,def(1))
        a2 = coords(:,def(2)) ! Central atom
        a3 = coords(:,def(3))

        ! Compute angle vectors
        call rel_vec(hmat,a2,a1,norm_vec=v1,dist=d1)
        call rel_vec(hmat,a2,a3,norm_vec=v2,dist=d2)

        ! Compute anlge axis
        ax = normalize(axis)

        ! Project angle vectors
        n1  = normalize(cross_prod(cross_prod(v1,ax),ax))
        n2  = normalize(cross_prod(cross_prod(v2,ax),ax))
        dn1 = dot_product(v1,n1)*d1
        dn2 = dot_product(v2,n2)*d2

        ! Compute angle value
        nn = cross_prod(n1,n2)
        val = -dot_product(n1,n2)
        val = min(max(val,-1._dp),1._dp)
        val = sign(1._dp,dot_product(nn,ax))*acos(val)

        ! Compute B-matrix indices
        ia1 = (/( 3*(def(1)-1)+i, i=1,3 )/)
        ia2 = (/( 3*(def(2)-1)+i, i=1,3 )/)
        ia3 = (/( 3*(def(3)-1)+i, i=1,3 )/)

        ! Compute B-matrix elements
        bmat(ia1) = -cross_prod(n1,ax)/dn1
        bmat(ia3) =  cross_prod(n2,ax)/dn2
        bmat(ia2) = -(bmat(ia1) + bmat(ia3))

    end subroutine
end module
