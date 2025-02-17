#include "shift_inverse_power_iteration.h"

namespace eigenvalue {

    using linear_algebra::operator*;
    using linear_algebra::operator-;

    double shift_inverse_power_iteration::solve(const linear_algebra::square_matrix& A, const double& mu, const std::vector<double>& x0) const
    {
        if (linear_algebra::norm(x0)!=1) {
            std::cerr << "Error: the input vector x0 should have norm equal to 1!" << std::endl;
            //return 0;
        }
        size_t N=A.size();
        linear_algebra::square_matrix A_new(N);

        // building the new matrix
        for (std::size_t j=0; j<N; ++j) {
            for(std::size_t i=0; i<N; ++i) {
                A_new(i,j)=A(i,j);
                if (i==j) {
                    A_new(i,j)=A_new(i,j)-mu;
                }
            }
        }
        // calling the inverse power iteration on the new matrix
        double lam=inverse_power_iteration::solve(A_new,x0);
        return mu+lam;
    }

} // eigenvalue