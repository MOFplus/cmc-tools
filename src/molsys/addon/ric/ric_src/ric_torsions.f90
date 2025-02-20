module ric_torsions

    use ric_common
    use ric_math

    implicit none

contains

    subroutine ric_torsions_bmat(hmat,coords,defs,ivals,ibrs,bmat,vals)
        real(dp), intent(in)    :: hmat(3,3)
        real(dp), intent(in)    :: coords(:,:)
        integer , intent(in)    :: defs(:,:)
        integer , intent(in)    :: ivals(:,:)
        integer , intent(in)    :: ibrs(:)
        real(dp), intent(inout) :: bmat(:,:)
        real(dp), intent(inout)   :: vals(:)

        integer :: ntorsion, i

        !f2py intent(in) hmat, coords, defs, ivals, ibrs
        !f2py intent(inout) bmat, vals

        ! Get the number of in-plane bends
        ntorsion = size(defs,dim=2)

        do i = 1, ntorsion
            call ric_torsion(hmat,coords,defs(:,i),ivals(:,i),bmat(:,ibrs(i)),vals(i))
        end do

    end subroutine

    ! This is from ff_gen_internals.f (torsio)
    subroutine ric_torsion(hmat,coords,def,ival,bmat,val)
        integer , parameter   :: max_atom      = 12
        integer , parameter   :: ib            = max_atom/2
        integer , parameter   :: ic            = ib + 1
        integer , parameter   :: ibs(max_atom) = (/ib,ib,ib,ib,ib,0,0,ic,ic,ic,ic,ic/)
        integer , parameter   :: ics(max_atom) = (/ic,ic,ic,ic,ic,0,0,ib,ib,ib,ib,ib/)

        real(dp), intent(in)    :: hmat(3,3)
        real(dp), intent(in)    :: coords(:,:)
        integer , intent(in)    :: def(max_atom)
        integer , intent(in)    :: ival(2)
        real(dp), intent(inout) :: bmat(:)
        real(dp), intent(out)   :: val

        integer  :: i, j, ia, iat(3,max_atom)
        real(dp) :: at(3,max_atom), axis(3), vec(3), d_axis, d_vec, c, s, f(3), &
            axes(3,max_atom), m(3), n(3), d_mn, c_mn(3)

        integer :: natoms(max_atom)

        at = huge(0._dp)
        forall(i = 1:max_atom, def(i) > 0)
            at(:,i) = coords(:,def(i))
            iat(:,i) = (/( 3*(def(i)-1)+j, j=1,3 )/)
        end forall

        call rel_vec(hmat,at(:,ib),at(:,ic),norm_vec=axis,dist=d_axis)
        axes(:,           1:max_atom/2) = spread( axis,2,max_atom/2)
        axes(:,max_atom/2+1:max_atom  ) = spread(-axis,2,max_atom/2)

        natoms = (/(count(def(1   :ib-1    ) > 0),i=1,max_atom/2), &
            (count(def(ic+1:max_atom) > 0),i=1,max_atom/2)/)

        ! Initialize the elements of centra atoms to 0,
        ! as their values are computed incrementally.
        bmat(iat(:,ib)) = 0._dp
        bmat(iat(:,ic)) = 0._dp

        do ia = 1, max_atom
            if (ia == ib .or. ia == ic .or. def(ia) == 0) cycle

            call rel_vec(hmat,at(:,ibs(ia)),at(:,ia),norm_vec=vec,dist=d_vec)
            c = dot_product(axes(:,ia),vec)
            !s = 1._dp - c**2; if (s <= 1e-10_dp) stop
            s = 1._dp - c**2
            f = cross_prod(axes(:,ia),vec)/(d_axis*d_vec*s*natoms(ia))

            bmat(iat(:,ia     )) =                      - f* d_axis
            bmat(iat(:,ibs(ia))) = bmat(iat(:,ibs(ia))) + f*(d_axis - d_vec*c)
            bmat(iat(:,ics(ia))) = bmat(iat(:,ics(ia))) + f*          d_vec*c

            ! Compute normal vectors
            if (ia ==    ival(1)) m = cross_prod(axis,vec); m = m/sqrt(sum(m*m))
            if (ia == ic+ival(2)) n = cross_prod(axis,vec); n = n/sqrt(sum(n*n))

        end do

        ! Campute dot and cross product of normal vectors
        d_mn = min(max(sum(m*n),-1._dp),1._dp)
        c_mn = -cross_prod(m,n)

        ! Compute dihedral angle
        val = sign(acos(d_mn),sum(axis*c_mn))

    end subroutine

end module
