#include "power_iteration.h"

namespace eigenvalue {

    using linear_algebra::operator*;
    using linear_algebra::operator-;

    double power_iteration::solve(const linear_algebra::square_matrix& A, const std::vector<double>& x0) const
    {
        std::size_t N=A.size();

        if (linear_algebra::norm(x0)!=1) {
            std::cerr << "Error: the input vector x0 should have norm equal to 1!" << std::endl;
        }
        std::size_t k=0;
        std::vector<double> x=x0;


        //std::vector<double> x_new;
        std::vector<double> aux;
        double nu0=0.;
        double nu1=0.;
        double residual=tolerance+1;
        double increment=tolerance+1;

        bool conv=false;

        while ( !conv && k < max_it) {

            // doing the scalar product of A with x0
            x=A*x;

            // normalizing the vector x_new
            linear_algebra::normalize(x);

            // computation of the eigenvalue
            aux=A*x;
            nu1=linear_algebra::scalar(aux,x);
            residual=linear_algebra::norm(aux-nu1*x);
            increment=std::abs(nu1-nu0)/std::abs(nu1);

            // Checking the convergence via the function converged
            conv=converged(residual, increment);

            // copying the new eigenvalue
            nu0=nu1;

            // updating the counter
            ++k;
        }
        // dealing the case of no convergence
        if (k==max_it) {
            std::cerr << "Error: the method does not converge in " << max_it <<" itaration!" << std::endl;
        }

        return nu0;
    }

    bool power_iteration::converged(const double& residual, const double& increment) const
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

} // eigenenvalue