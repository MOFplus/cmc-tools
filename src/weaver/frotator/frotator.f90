subroutine is_superpose(n,v1,v2,thresh,tof,rmsd)
    implicit none
    integer, intent(in)                 ::  n
    real(8), intent(in)                 ::  v1(3,n), v2(3,n),thresh
    real(8), intent(out)                ::  rmsd
!     integer, intent(out)                ::  tof
    logical, intent(out)                ::  tof
    integer                             ::  i,j
    real(8)                             ::  d,r(3),dmin,ddot
    !f2py intent(in) n
    !f2py depend(n) v1
    !f2py depend(n) v2
    !f2py intent(in) thresh
    !f2py intent(out) tof
    !f2py intent(out) rmsd
    
    rmsd = 0.0d0
    tof = .true.
    do i=1,n
        dmin=100000.0d0
        do j=1,n
            r = v2(:,i) - v1(:,j)
            d = sqrt(r(1)*r(1) + r(2)*r(2) + r(3)*r(3))
            if (dmin.gt.d) then
                dmin = d
            end if 
        end do
        if (dmin.gt.thresh) then
            tof = .false.
!             rmsd = dmin
!             exit
        end if
        rmsd = rmsd + dmin * dmin
    end do
    rmsd = sqrt(rmsd) / float(n)

end subroutine is_superpose

subroutine calc_acos_mat(n,v1,v2,acos_mat)
!     use omp_lib
    implicit none
    integer, intent(in)                 ::  n
    real(8), intent(in)                 ::  v1(3,n), v2(3,n)
    real(8), intent(out)                ::  acos_mat(n,n)
    integer                             ::  i,j
!f2py intent(in) n
!f2py depend(n) v1
!f2py depend(n) v2
!f2py depend(n) cosa_mat
!f2py intent(out) acos_mat

    do i=1,n
        do j=1,n
             acos_mat(i,j) = 1.0d0-(v1(1,i)*v2(1,j)+v1(2,i)*v2(2,j)+v1(3,i)*v2(3,j))
        end do
    end do
    
end subroutine calc_acos_mat

subroutine get_rotmat(triple,R)
!     use blas
    implicit none
    real(8), intent(in)                 ::  triple(3)
    real(8), intent(out)                ::  R(3,3)
    real(8)                             ::  t(3),rr(2),quat(4),nn
    real(8),parameter                   ::  pi = 3.141592653589793238462643d0
    
!f2py intent(in) triple
!f2py intent(out) R

    ! 1: get rotation quaternion:  make sure 0<t[0]<1
    t(1) =  triple(1)
    if (t(1).lt.0.0d0) t(1) = 0.0d0 
    if (t(1).gt.1.0d0) t(1) = 1.0d0 
    rr(1) = sqrt(1.0d0-t(1))
    rr(2) = sqrt(t(1))
    t(2) = 2.0d0 * pi * mod(triple(2),1.0d0)
    t(3) = 2.0d0 * pi * mod(triple(3),1.0d0)
    quat(:) = (/ cos(t(3))*rr(2) , sin(t(2))*rr(1) , cos(t(2))*rr(1) , sin(t(3))*rr(2) /)
    nn = quat(1)*quat(1)+quat(2)*quat(2)+quat(3)*quat(3)+quat(4)*quat(4)
    if (nn .lt. 1.0e-14) then
        R(:,1) = (/1.0d0, 0.0d0,0.0d0/)
        R(:,2) = (/0.0d0, 1.0d0,0.0d0/)
        R(:,3) = (/0.0d0, 0.0d0,1.0d0/)
    else
        quat = quat * sqrt(2.0d0 / nn)
        R(1,1) = 1.0d0 - quat(3)*quat(3)-quat(4)*quat(4)
        R(1,2) =         quat(2)*quat(3)+quat(4)*quat(1)
        R(1,3) =         quat(2)*quat(4)-quat(3)*quat(1)
        R(2,1) =         quat(2)*quat(3)-quat(4)*quat(1)
        R(2,2) = 1.0d0 - quat(2)*quat(2)-quat(4)*quat(4)
        R(2,3) =         quat(3)*quat(4)+quat(2)*quat(1)
        R(3,1) =         quat(2)*quat(4)+quat(3)*quat(1)
        R(3,2) =         quat(3)*quat(4)-quat(2)*quat(1)
        R(3,3) = 1.0d0 - quat(2)*quat(2)-quat(3)*quat(3)
    end if

end subroutine get_rotmat


! 
subroutine calc_penalty(n,v1,v2,which,triple,pen)
    implicit none
    
    integer, intent(in)         :: n, which(n)
    real(8), intent(in)         :: v1(3,n), v2(3,n),triple(3)
    real(8), intent(out)        :: pen
    
    real(8)                     :: x2(3,n),ddot, val
    real(8),parameter           ::  pi = 3.141592653589793238462643d0
    integer                     :: i
    
!f2py intent(in) n
!f2py intent(in) triple
! ! !f2py depend(n) x2
!f2py depend(n) v1
!f2py depend(n) v2
!f2py depend(n) which
!f2py intent(out) pen

    x2 = v2
    call rotate_by_triple(n,x2,triple)
    pen=0.0d0
    do i=1,n
!         pen = pen + (1.0d0-ddot(3,v1(:,i),1,x2(:,which(i)+1),1))
!         pen = pen + (1.0d0-acos(ddot(3,v1(:,i),1,x2(:,which(i)+1),1))/pi)
!         pen = pen + acos(ddot(3,v1(:,i),1,x2(:,which(i)+1),1))/pi
        val = v1(1,i)*x2(1,which(i)+1) + v1(2,i)*x2(2,which(i)+1) + v1(3,i)*x2(3,which(i)+1)
        val = min(max(val, -1.0d0), 1.0d0)
        pen = pen + ( acos( val ) / pi * 180.0d0)
!         pen = pen + ( acos( v1(1,i)*x2(1,which(i)+1) + v1(2,i)*x2(2,which(i)+1) + v1(3,i)*x2(3,which(i)+1) ) / pi * 180.0d0)**2.0d0
!         pen = pen + (1.0d0-ddot(3,v1(:,i),1,x2(:,which(i)+1),1))
!         pen = pen + (1.0d0-acos(ddot(3,v1(:,i),1,x2(:,which(i)+1),1))/pi)
    end do
    pen = pen / float(n)
    ! pen = sqrt(pen) / float(n)

end subroutine calc_penalty

subroutine calc_penaltysq(n,v1,v2,which,triple,pen)
    implicit none
    
    integer, intent(in)         :: n, which(n)
    real(8), intent(in)         :: v1(3,n), v2(3,n),triple(3)
    real(8), intent(out)        :: pen
    
    real(8)                     :: x2(3,n),ddot, val
    real(8),parameter           ::  pi = 3.141592653589793238462643d0
    integer                     :: i
    
!f2py intent(in) n
!f2py intent(in) triple
! ! !f2py depend(n) x2
!f2py depend(n) v1
!f2py depend(n) v2
!f2py depend(n) which
!f2py intent(out) pen

    x2 = v2
    call rotate_by_triple(n,x2,triple)
    pen=0.0d0
    do i=1,n
        val = v1(1,i)*x2(1,which(i)+1) + v1(2,i)*x2(2,which(i)+1) + v1(3,i)*x2(3,which(i)+1)
        val = min(max(val, -1.0d0), 1.0d0)
        pen = pen + ( acos( val ) / pi * 180.0d0)**2.0d0
    end do
    pen = sqrt(pen) / float(n)

end subroutine calc_penaltysq


subroutine grad(n,v1,v2,which,triple,delt,pen,gradient)
    ! what we want to do here is a single sided gradient depending on where the actual value of t[i] is.
    implicit none
    
    integer, intent(in)         :: n, which(n)
    real(8), intent(in)         :: v1(3,n), v2(3,n),triple(3),delt
    real(8), intent(out)        :: gradient(3),pen
    
    real(8)                     :: x2(3,n),ddot,tt(3),p(3)
    real(8),parameter           :: pi = 3.141592653589793238462643d0
    integer                     :: i
    
!f2py intent(in) n
!f2py intent(in) triple,delt
! ! !f2py depend(n) x2
!f2py depend(n) v1
!f2py depend(n) v2
!f2py depend(n) which
!f2py intent(out) gradient,pen

    call calc_penalty(n,v1,v2,which,triple,pen)
    do i =1,3
        tt=triple
        if (nint(triple(i)).eq.1) then     ! left-sided gradient!
            tt(i) = triple(i) - delt
            call calc_penalty(n,v1,v2,which,tt,p(i)) ! stores penalty in p(i)
            gradient(i) = (pen-p(i)) / delt
        else                               ! right-sided gradient!
            tt(i) = triple(i) + delt
            call calc_penalty(n,v1,v2,which,tt,p(i)) ! stores penalty in p(i)
            gradient(i) = (p(i)-pen) / delt
        end if
    
    end do
end subroutine grad

subroutine rotate_by_triple(n, xyz, triple)
!     use blas
    implicit none
    integer, intent(in)                 ::  n
    real(8), intent(in)                 ::  triple(3)
    real(8), intent(inout)              ::  xyz(3,n)
    integer                             ::  i,j
    real(8)                             ::  t(3),rr(2),quat(4), R(3,3),nn,txyz(3)
    real(8),parameter                   ::  pi = 3.141592653589793238462643d0
    
!f2py intent(in) n
!f2py intent(in) triple
!f2py depend(n) xyz
!f2py intent(inout) xyz

    ! 1: get rotation quaternion:  make sure 0<t[0]<1
    t(1) =  triple(1)
    if (t(1).lt.0.0d0) t(1) = 0.0d0 
    if (t(1).gt.1.0d0) t(1) = 1.0d0 
    rr(1) = sqrt(1.0d0-t(1))
    rr(2) = sqrt(t(1))
    t(2) = 2.0d0 * pi * mod(triple(2),1.0d0)
    t(3) = 2.0d0 * pi * mod(triple(3),1.0d0)
    quat(:) = (/ cos(t(3))*rr(2) , sin(t(2))*rr(1) , cos(t(2))*rr(1) , sin(t(3))*rr(2) /)
    nn = quat(1)*quat(1)+quat(2)*quat(2)+quat(3)*quat(3)+quat(4)*quat(4)
    if (nn .lt. 1.0e-14) then
        R(:,1) = (/1.0d0, 0.0d0,0.0d0/)
        R(:,2) = (/0.0d0, 1.0d0,0.0d0/)
        R(:,3) = (/0.0d0, 0.0d0,1.0d0/)
    else
        quat = quat * sqrt(2.0d0 / nn)
        R(1,1) = 1.0d0 - quat(3)*quat(3)-quat(4)*quat(4)
        R(1,2) =         quat(2)*quat(3)+quat(4)*quat(1)
        R(1,3) =         quat(2)*quat(4)-quat(3)*quat(1)
        R(2,1) =         quat(2)*quat(3)-quat(4)*quat(1)
        R(2,2) = 1.0d0 - quat(2)*quat(2)-quat(4)*quat(4)
        R(2,3) =         quat(3)*quat(4)+quat(2)*quat(1)
        R(3,1) =         quat(2)*quat(4)+quat(3)*quat(1)
        R(3,2) =         quat(3)*quat(4)-quat(2)*quat(1)
        R(3,3) = 1.0d0 - quat(2)*quat(2)-quat(3)*quat(3)
    end if
    do i=1,n
        call dgemv('n',3,3,1.0d0,R,3,xyz(:,i),1,0.0d0,txyz,1) ! matrix vector product
        xyz(:,i) = txyz(:)
    end do

end subroutine rotate_by_triple

!!!!! routines for the topoFF rotator


subroutine xyzgrad1(n,v1,v2,which,triple,delt,pen,gradient)
    ! here we want to have the gradient of the actual coordinates instead of the orientation values, will be 3N?!
    implicit none
    
    integer, intent(in)         :: n, which(n)
    real(8), intent(in)         :: v1(3,n), v2(3,n),triple(3),delt
    real(8), intent(out)        :: gradient(3,n),pen
    
    real(8)                     :: x2(3,n),x3(3,n),tt(3),p(3),pen0,pp,pm
    real(8),parameter           :: pi = 3.141592653589793238462643d0
    integer                     :: i,j
    
!f2py intent(in) n
!f2py intent(in) triple,delt
!f2py depend(n) v1
!f2py depend(n) v2
!f2py depend(n) which
!f2py depend(n) gradient
!f2py intent(out) gradient,pen

    x2 = v2
    call rotate_by_triple(n,x2,triple)
    call calc_penalty_fixed(n,v1,x2,which,pen)
    x3 = x2
    do i=1,n ! loop over natoms
        do j=1,3   ! loop over x,y,z
            x3(:,i) = x2(:,i)
            x3(j,i) = x2(j,i) + delt
            x3(:,i) =  x3(:,i) / sqrt(x3(1,i)*x3(1,i) + x3(2,i)*x3(2,i) + x3(3,i)*x3(3,i))
            call calc_penalty_fixed(n,v1,x3,which,pp)
            x3(:,i) = x2(:,i)
            x3(j,i) = x2(j,i) - delt
            x3(:,i) =  x3(:,i) / sqrt(x3(1,i)*x3(1,i) + x3(2,i)*x3(2,i) + x3(3,i)*x3(3,i))
            call calc_penalty_fixed(n,v1,x3,which,pm)
            gradient(j,i) = (pp-pm)  /  (2.0d0*delt) * pen
            x3(:,i) = x2(:,i)
        end do
    end do !i,j
end subroutine xyzgrad1


subroutine xyzgrad2(n,v1,v2,which,triple,delt,pen,gradient)
    ! here we want to have the gradient of the actual coordinates instead of the orientation values, will be 3N?!
    implicit none
    
    integer, intent(in)         :: n, which(n)
    real(8), intent(in)         :: v1(3,n), v2(3,n),triple(3),delt
    real(8), intent(out)        :: gradient(3,n),pen
    
    real(8)                     :: x2(3,n),x1(3,n),tt(3),p(3),pen0,pp,pm
    real(8),parameter           :: pi = 3.141592653589793238462643d0
    integer                     :: i,j
    
!f2py intent(in) n
!f2py intent(in) triple,delt
!f2py depend(n) v1
!f2py depend(n) v2
!f2py depend(n) which
!f2py depend(n) gradient
!f2py intent(out) gradient,pen

    x2 = v2
    x1 = v1
    call rotate_by_triple(n,x2,triple)
    call calc_penaltysq_fixed(n,v1,x2,which,pen)
    do i=1,n ! loop over natoms
        do j=1,3   ! loop over x,y,z
            x1(:,i) = v1(:,i)
            x1(j,i) = v1(j,i) + delt
            x1(:,i) =  x1(:,i) / sqrt(x1(1,i)*x1(1,i) + x1(2,i)*x1(2,i) + x1(3,i)*x1(3,i))
            call calc_penaltysq_fixed(n,x1,x2,which,pp)
            x1(:,i) = v1(:,i)
            x1(j,i) = v1(j,i) - delt
            x1(:,i) =  x1(:,i) / sqrt(x1(1,i)*x1(1,i) + x1(2,i)*x1(2,i) + x1(3,i)*x1(3,i))
            call calc_penaltysq_fixed(n,x1,x2,which,pm)
            gradient(j,i) = (pp-pm)  /  (2.0d0*delt) ! * pen
            x1(:,i) = v1(:,i)
        end do
    end do
end subroutine xyzgrad2


subroutine calc_penaltysq_fixed(n,v1,v2,which,pen)
    implicit none
    
    integer, intent(in)         :: n, which(n)
    real(8), intent(in)         :: v1(3,n), v2(3,n)
    real(8), intent(out)        :: pen
    
    real(8)                     :: ddot,dd
    real(8),parameter           :: pi = 3.141592653589793238462643d0
    integer                     :: i
    
!f2py intent(in) n
! ! !f2py depend(n) x2
!f2py depend(n) v1
!f2py depend(n) v2
!f2py depend(n) which
!f2py intent(out) pen

    pen=0.0d0
    do i=1,n
        dd = v1(1,i)*v2(1,which(i)+1) + v1(2,i)*v2(2,which(i)+1) + v1(3,i)*v2(3,which(i)+1)
        if (dd.le.0.9999999999d0) then
            pen = pen + (acos(dd) / pi * 180.0d0) ** 2.0d0
        end if
    end do
    pen = pen / float(n)

end subroutine calc_penaltysq_fixed


subroutine calc_penalty_fixed(n,v1,v2,which,pen)
    implicit none
    
    integer, intent(in)         :: n, which(n)
    real(8), intent(in)         :: v1(3,n), v2(3,n)
    real(8), intent(out)        :: pen
    
    real(8)                     :: ddot,dd
    real(8),parameter           :: pi = 3.141592653589793238462643d0
    integer                     :: i
    
!f2py intent(in) n
! ! !f2py depend(n) x2
!f2py depend(n) v1
!f2py depend(n) v2
!f2py depend(n) which
!f2py intent(out) pen

    pen=0.0d0
    do i=1,n
        dd = v1(1,i)*v2(1,which(i)+1) + v1(2,i)*v2(2,which(i)+1) + v1(3,i)*v2(3,which(i)+1)
        if (dd.le.0.9999999999d0) then
            pen = pen + (acos(dd) / pi * 180.0d0)
        end if
    end do
    pen = pen / float(n)
end subroutine calc_penalty_fixed



    

