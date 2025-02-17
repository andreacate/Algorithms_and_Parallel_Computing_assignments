#include "convolutional_layer.hpp"

namespace convnet {

    convolutional_layer::convolutional_layer(std::size_t _s_filter, std::size_t _prev_depth, std::size_t _n_filters,
                                             std::size_t _s_stride, std::size_t _s_padding)
            : s_filter(_s_filter), prev_depth(_prev_depth), n_filters(_n_filters), s_stride(_s_stride),
              s_padding(_s_padding) {
        initialize();
    }

    void convolutional_layer::initialize() {
        for (std::size_t it = 0; it < n_filters; ++it) {
            tensor_3d filter(s_filter, s_filter, prev_depth);
            filter.initialize_with_random_normal(0.0, 3.0 / (2 * s_filter + prev_depth));
            filters.push_back(filter);
        }
    }

    tensor_3d convolutional_layer::evaluate(const tensor_3d &inputs) const {


        //Output tensor dimension
        const std::size_t H_out = (inputs.get_height()-s_filter+2*s_padding) / s_stride + 1;
        const std::size_t W_out = (inputs.get_width()-s_filter+2*s_padding) / s_stride + 1;

        // Output tensor
        tensor_3d ret(H_out,W_out,n_filters);
        ret.initialize_with_zeros();


        if ( prev_depth != inputs.get_depth() || prev_depth != filters[0].get_depth() || inputs.get_depth() != filters[0].get_depth() ){
            std::cerr << "The number of channel in the input are not the same number of filters' channels" << std::endl;
            return ret;
        }


        double aux=0.0;
        // cycling through the number of filters
        for(std::size_t hh=0; hh < n_filters; ++hh) {
            //cycling through the rows of the input
            for(std::size_t ii=0; ii < inputs.get_height()-(s_filter-1); ++ii) {
                //cycling through the cols of the input
                for(std::size_t jj=0; jj < inputs.get_width()-(s_filter-1); ++jj){
                    //cycling thorough the number of channels
                    for(std::size_t k=0; k < prev_depth; ++k) {
                        aux=0.0;
                        //cycling thorough rows of the filter
                        for(std::size_t i=0; i < s_filter; ++i) {
                            //cycling though cols of the filter
                            for(std::size_t j=0; j < s_filter; ++j) {
                                aux += inputs(ii+i,jj+j,k)*filters[hh](i,j,k);
                            }
                        }
                        ret(ii/s_stride,jj/s_stride,hh)+=aux;
                    }
                    jj += (s_stride-1);
                }
                ii += (s_stride-1);

            }
        }
        return ret;

    }

    tensor_3d convolutional_layer::apply_activation(const tensor_3d &Z) const {
        return act_function.apply(Z);
    }

    tensor_3d convolutional_layer::forward_pass(const tensor_3d &inputs) const {


        tensor_3d res=evaluate(inputs);
        return apply_activation(res);
    }


    std::vector<std::vector<double>> convolutional_layer::get_parameters() const {
        std::vector<std::vector<double> > parameters;
        for (tensor_3d filter: filters) {
            parameters.push_back(filter.get_values());
        }
        return parameters;
    }

    void convolutional_layer::set_parameters(const std::vector<std::vector<double>> parameters) {
        for (std::size_t i = 0; i < n_filters; ++i) {
            filters[i].set_values(parameters[i]);
        }
    }

} // namespace