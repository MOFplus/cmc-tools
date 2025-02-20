module ric_eckart

    use ric_common
    use ric_math

    implicit none

contains

    subroutine ric_eckart_trans_bmat(coords,masses,defs,ibrs,bmat,vals)
        real(dp), intent(in)    :: coords(:,:)
        real(dp), intent(in)    :: masses(:)
        integer , intent(in)    :: defs(:)
        integer , intent(in)    :: ibrs(:)
        real(dp), intent(inout) :: bmat(:,:)
        real(dp), intent(inout) :: vals(:)

        integer  :: i
        real(dp) :: fact

        !f2py intent(in) coords, masses, defs, ibrs
        !f2py intent(inout) bmat, vals

        ! Compute B-matrix elements for translation Eckart coordinates
        fact = 1._dp/(sum(masses)*size(masses))
        forall(i=1:size(defs)) &
                bmat(defs(i)::3,ibrs(i)) = fact*masses
            !bmat(i::3,ibrs(i)) = masses/sum(masses)

            ! Compute values for translation Eckart coordinates
            ! Effectively it is a centre of mass
            !fact = 1._dp/sum(masses)
            forall(i=1:size(defs)) vals(i) = fact*sum(coords(defs(i),:)*masses)
    end subroutine

    subroutine ric_eckart_rot_bmat(coords,masses,defs,ibrs,bmat,vals)
        real(dp), intent(in)    :: coords(:,:)
        real(dp), intent(in)    :: masses(:)
        integer , intent(in)    :: defs(:)
        integer , intent(in)    :: ibrs(:)
        real(dp), intent(inout) :: bmat(:,:)
        real(dp), intent(inout) :: vals(:)

        ! Construct constant 3x3 array, where columns are Cartesian basis vectors
        real(dp), parameter :: axes_(9) = (/ 1._dp, 0._dp, 0._dp, &
            0._dp, 1._dp, 0._dp, &
            0._dp, 0._dp, 1._dp /)
        real(dp), parameter :: axes(3,3) = reshape(axes_,(/3,3/))

        integer  :: i, j
        real(dp) :: cent(3), fact

        !f2py intent(in) coords, masses, defs, ibrs
        !f2py intent(inout) bmat, vals
        ! Compute the centre of mass
        fact = 1._dp/sum(masses)
        forall(i=1:3) cent(i) = fact*sum(coords(i,:)*masses)

            ! Compute B-matrix elements for rotational Eckart coordinates
            fact = 1._dp/(sum(masses)*size(masses))
            forall(i=1:size(defs),j=1:size(masses))
                bmat(3*(j-1)+1:3*(j-1)+3,ibrs(i)) = fact*masses(j) &
                    * cross_prod(axes(:,defs(i)),coords(:,j) - cent)
                !bmat(3*(j-1)+1:3*(j-1)+3,ibrs(i)) = masses(j)/size(masses) &
                !                                   * cross_prod(axes(:,defs(i)),coords(:,j) - cent )
            end forall

            ! Compute values for rotation Eckart coordinates
            forall(i=1:size(defs)) vals(i+3) = huge(0._dp) ! Not implemented

    end subroutine
end module
