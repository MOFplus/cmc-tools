module red_int_coords

  use ric_common
  implicit none

contains

  subroutine bmat_invert(bmat, bmat_inv, rank, info)

    integer :: nric, ncart, i, nwork, info, rank 
    real(dp), allocatable :: bmat_(:,:), rhs(:,:), sing(:), work(:)
    real(dp), intent(in)  :: bmat(:,:)
    real(dp), intent(inout) :: bmat_inv(:,:)

    !f2py intent(in) bmat
    !f2py intent(inout) bmat_inv
    !f2py intent(out) rank, info

    ! stuff from ff_gen_b_invert.f

!     Make a copy of B-matrix, since dgelss overwrites it.
!     A = b

    ! Get numbers
    ncart = size(bmat,dim=1) ! # of redundant internat coordinates
    nric  = size(bmat,dim=2) ! # of Cartesian coordinates

    ! Make a copy of B-matrix, since dgelss overwrites it
    allocate(bmat_(nric,ncart))
    bmat_ = transpose(bmat) ! HACK

!     Construct the left-hand-side matrix as a diagonal matrix
!      C = 0.0d0
!      do i = 1,ir
!        C(i,i) = 1.0d0
!      end do

    ! Construct right hand side matrix
    allocate(rhs(nric,nric))
    rhs = 0._dp
    forall(i=1:nric) rhs(i,i) = 1._dp

    ! Allocate singular value and work arrays
    allocate(sing(ncart))
    nwork = 3*ncart + max(2*ncart,nric)
    allocate(work(nwork))

!     Invert B-matrix.
!      call dgelss(ir,k,ir,A,ms,C,ms,S,1.0e-10,rank,work,ms*ms,
!      *           info)

!   Compute pseudo inverse of B-matrix
    call dgelss(nric  , & ! M
                ncart , & ! N
                nric  , & ! NRHS = M
                bmat_ , & ! A(LDA,N) = A(M,N)
                nric  , & ! LDA = M
                rhs   , & ! B(LDB,NRHS) = B(M,M)
                nric  , & ! LDB = M
                sing  , & ! S(N)
                -1._dp, & ! RCOND
                rank  , & ! RANK
                work  , & ! WORK(NWORK)
                nwork , & ! NWORK
                info    ) ! INFO
    bmat_inv = rhs(1:ncart,:)

!     Inverted and transposed B-matrix is in the vv array
!      vv(1:ir,1:k) = transpose(C(1:k,1:ir))

!      if (info .eq. 0) then
!        write(iout,'(a,/)') 'done!'
!      else
!        write(iout,'(a,i5,/)') 'dgelss error: ', info
!        idm = 0
!        return
!      end if

!      Check if B-matrix is rank-defincient
!       if (rank .lt. ir) idm = 0

!    bmat_invert = info

  end subroutine

  subroutine hessian_project(bmat, bmat_inv, cart_hessian, ric_hessian, stat)
    
    integer :: nric, ncart, stat
    real(dp), allocatable :: tmp(:,:)
    real(dp), intent(in) :: bmat(:,:), bmat_inv(:,:), cart_hessian(:,:)
    real(dp), intent(inout) :: ric_hessian(:,:)


    !f2py intent(in) bmat, bmat_inv, cart_hessian
    !f2py intent(inout) ric_hessian
    !f2py intent(out) stat


    ! stuff from ff_gen_trans_hess.f

    ! Get numbers
    ncart = size(bmat,dim=1) ! # of redundant internat coordinates
    nric  = size(bmat,dim=2) ! # of Cartesian coordinates

    ! Allocate a temporary array
    allocate(tmp(ncart,nric),stat=stat)


!     Multiply the Hessian matrix and the inverted B-matrix
!     taking into accout that the former is symmetrix.
!      call dsymm('L','U',ir,k,1.0d0,v,ms,vv,ms,0.0d0,tmp,ms)

!     Multilpy the transposed inverted B-matrix and the matrix
!     from previous operations.
!      call dgemm('T','N',k,k,ir,1.0d0,vv,ms,tmp,ms,0.0d0,v,ms)

!   Multiply Cartesian Hessian matrix and the B-matrix
!   taking into accout that the former is symmetrix
    call dsymm('L'         , & ! SIDE
               'U'         , & ! UPLO
               ncart       , & ! M
               nric        , & ! N
               1._dp       , & ! ALPHA
               cart_hessian, & ! A(LDA,ka) = A(M,M)
               ncart       , & ! LDA = M
               bmat_inv    , & ! B(LDB,N) = B(M,N)
               ncart       , & ! LDB = M
               0._dp       , & ! BETA
               tmp         , & ! C(LDC,N) = C(M,N)
               ncart         ) ! LDC = M

!   Multilpy the transposed inverted B-matrix and the matrix
!   from previous operations
    call dgemm('T'        , & ! TRANSA
               'N'        , & ! TRANSB
               nric       , & ! M
               nric       , & ! N
               ncart      , & ! K
               1._dp      , & ! ALPHA
               bmat_inv   , & ! A(LDA,ka) = A(K,M)
               ncart      , & ! LDA = K
               tmp        , & ! B(LDB,kb) = B(K,N)
               ncart      , & ! LDB = K
               0._dp      , & ! BETA
               ric_hessian, & ! C(LDC,N) = C(M,N)
               nric         ) ! LDC = M

!    hessian_project = stat

  end subroutine

end module

