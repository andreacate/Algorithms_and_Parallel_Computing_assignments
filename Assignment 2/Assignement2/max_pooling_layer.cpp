#include "max_pooling_layer.hpp"

namespace convnet {

    max_pooling_layer::max_pooling_layer(std::size_t s_filter, std::size_t strd) {
        size_filter = s_filter;
        stride = strd;
    };

    tensor_3d max_pooling_layer::evaluate(const tensor_3d &inputs) const {

        // Computing the dimensions of the output tensor
        const std::size_t H_out=(inputs.get_height()-size_filter)/stride+1;
        const std::size_t W_out=(inputs.get_width()-size_filter)/stride+1;
        const std::size_t D=inputs.get_depth();

        tensor_3d ret(H_out,W_out,D);
        ret.initialize_with_zeros();

        double aux;

        for(std::size_t k=0; k < D; ++k) {
            for(std::size_t ii=0; ii < inputs.get_height() - (size_filter - 1); ++ii) {
                for(std::size_t jj=0; jj < inputs.get_width() - (size_filter - 1); ++jj) {
                    aux = inputs(ii,jj,k);
                    for( std::size_t i=0; i < size_filter; ++i ) {
                        for( std::size_t j=0; j < size_filter; ++j ) {
                            if( inputs(ii+i,jj+j,k)>aux )
                                aux = inputs(ii+i,jj+j,k);
                        }
                    }
                    ret(ii/stride,jj/stride,k)=aux;
                    jj += (stride-1);
                }
                ii += (stride-1);
            }
        }
        return ret;

    };


    tensor_3d max_pooling_layer::apply_activation(const tensor_3d &Z) const {
        return Z;
    };

    tensor_3d max_pooling_layer::forward_pass(const tensor_3d &inputs) const {

        tensor_3d res=evaluate(inputs);
        return apply_activation(res);

    };

    // Do nothing since max pooling has no learnable parameter
    void max_pooling_layer::set_parameters(const std::vector<std::vector<double>> parameters) {}

} // namespace