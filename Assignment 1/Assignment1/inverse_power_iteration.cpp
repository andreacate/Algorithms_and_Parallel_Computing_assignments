#include "inverse_power_iteration.h"

#include "power_iteration.h"


namespace eigenvalue {

    using linear_algebra::operator*;
    using linear_algebra::operator-;

    double inverse_power_iteration::solve(const linear_algebra::square_matrix& A, const std::vector<double>& x0) const
    {
        if (linear_algebra::norm(x0)!=1) {
            std::cerr << "Error: the input vector x0 should have norm equal to 1!" << std::endl;
        }
        size_t N=A.size();
        linear_algebra::square_matrix L(N);
        linear_algebra::square_matrix U(N);
        linear_algebra::square_matrix L_inv(N);
        linear_algebra::square_matrix U_inv(N);
        linear_algebra::square_matrix A_inv(N);
        linear_algebra::lu(A,L,U);

        std::vector<double> y(N);

        // inverting the matrix U (upper triangular)
        for( size_t i=0; i< N; ++i) {
            std::vector<double> b(N,0.);
            b[i]=1.;
            y=linear_algebra::backsolve(U,b);
            for (size_t jj=0; jj<N; ++jj) {
                U_inv(jj,i)=y[jj];
            }
        }

        // auxiliary vector to store column of the matrix inverse of A
        std::vector<double> A_inv_col(N);
        std::vector<double> L_inv_col(N);

        // for cycle to invert the matrix L and building directly the matrix inverse of A
        for( size_t jj=0; jj< N; ++jj) {
            std::vector<double> b(N,0.);
            b[jj]=1.;
            L_inv_col=linear_algebra::forwardsolve(L,b);

            // finding the column of the matrix inverse of A, using the scalar product of matrix U_inv and L_inv_column
            A_inv_col = U_inv * L_inv_col;
            // building the matrix inverse of A
            for (size_t ii=0; ii < N; ii++) {
                A_inv(ii,jj) = A_inv_col[ii];
            }
        }

        // recalling power iteration to the matrix inverse of A
        eigenvalue::power_iteration P(max_it, tolerance, termination);
        double nu_inv=P.solve(A_inv,x0);

        if (nu_inv==0) {
            std::cerr << "Error divinding by 0" << std::endl;
        }

        return 1/nu_inv;
    }

    bool inverse_power_iteration::converged(const double& residual, const double& increment) const
    {
        bool conv;

        switch(termination) {
            case(RESIDUAL):
                conv = residual < tolerance;
                break;
            case(INCREMENT):
                conv = increment < tolerance;
                break;
            case(BOTH):
                conv = residual < tolerance && increment < tolerance;
                break;
            default:
                conv = false;
        }
        return conv;
    }

} // e eigenvalue