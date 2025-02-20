module ric_out_bends

    use ric_common
    use ric_math

    implicit none

contains


    subroutine ric_out_bends_bmat(hmat,coords,defs,ibrs,bmat,vals)
        real(dp), intent(in)    :: hmat(3,3)
        real(dp), intent(in)    :: coords(:,:)
        integer , intent(in)    :: defs(:,:)
        integer , intent(in)    :: ibrs(:)
        real(dp), intent(inout) :: bmat(:,:)
        real(dp), intent(inout) :: vals(:)

        integer :: nout_bend, i

        !f2py intent(in) hmat, coords, defs, ibrs
        !f2py intent(inout) bmat, vals

        ! Get the number of out-of-plane bends
        nout_bend = size(defs,dim=2)

        do i = 1, nout_bend
            call ric_out_bend(hmat,coords,defs(:,i),bmat(:,ibrs(i)),vals(i))
        end do

    end subroutine

    ! This is from ff_gen_internals.f (wagin4)
    subroutine ric_out_bend(hmat,coords,def,bmat,val)
        real(dp), intent(in)    :: hmat(3,3)
        real(dp), intent(in)    :: coords(:,:)
        integer , intent(in)    :: def(4)
        real(dp), intent(inout) :: bmat(:)
        real(dp), intent(out)   :: val


        integer  :: i, ia1(3), ia2(3), ia3(3), ia4(3)
        real(dp) :: a1(3) , a2(3) , a3(3) , a4(3) , cc(3), &
            a1_(3), a2_(3), a3_(3), &
            e41(3) , e42(3) , e43(3) , os23(3), &
            e41_(3), e42_(3), &
            pn, r41 , r42, r43, r41_, r1, ss, c, s

        ! Get atom positions
        a1 = coords(:,def(2))
        a2 = coords(:,def(3))
        a3 = coords(:,def(4))
        a4 = coords(:,def(1)) ! Central atom

        call rel_vec(hmat,a4,a1,norm_vec=e41,dist=r41)
        call rel_vec(hmat,a4,a2,norm_vec=e42,dist=r42)
        call rel_vec(hmat,a4,a3,norm_vec=e43,dist=r43)

        pn   = sqrt(max(1._dp - cdc(hmat,a3,a4,a2)**2,0._dp))
        os23 = cross_prod(e42,e43)
        ss   = sum(e41*os23)
        val  = acos(ss/pn)
        cc   = e41 - os23*ss/pn**2 + a4

        call rel_vec(hmat,a4,cc,norm_vec=e41_,dist=r41_)

        a1_ = cross_prod(e41_,e42 )
        a2_ = cross_prod(e43 ,e41_)
        a3_ = a4 + os23

        call rel_vec(hmat,a4,a1 ,norm_vec=e41_,dist=r1)
        call rel_vec(hmat,a4,a3_,norm_vec=e42_)
        c = cdc(hmat,a1,a4,a3_)
        s = sqrt(max(1._dp - c**2,0._dp))

        ! Compute B-matrix indices
        ia1 = (/( 3*(def(2)-1)+i, i=1,3 )/)
        ia2 = (/( 3*(def(3)-1)+i, i=1,3 )/)
        ia3 = (/( 3*(def(4)-1)+i, i=1,3 )/)
        ia4 = (/( 3*(def(1)-1)+i, i=1,3 )/)

        ! Compute B-matrix elements
        bmat(ia1) = (c*e41_ - e42_)/(r1*s)
        bmat(ia2) = -a2_/(r42*pn)
        bmat(ia3) = -a1_/(r43*pn)
        bmat(ia4) = -bmat(ia1) - bmat(ia2) - bmat(ia3)

    end subroutine
end module
