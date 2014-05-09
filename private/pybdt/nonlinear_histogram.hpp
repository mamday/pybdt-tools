// nonlinear_histogram.hpp


#ifndef PYBDT_NONLINEAR_HISTOGRAM_HPP
#define PYBDT_NONLINEAR_HISTOGRAM_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <ostream>
#include <vector>
#include <utility>

#include "histogram.hpp"

class NonlinearHistogram : public Histogram {
public:

    // explicit Histogram (const vector<double>& values);
    explicit NonlinearHistogram (const std::vector<double>& bin_edges);
    NonlinearHistogram (const int n,
                        const std::vector<double> values,
                        bool presorted=false);
    NonlinearHistogram (const int n,
                        const std::vector<double> values,
                        const std::vector<double> weights,
                        bool presorted=false);
    ~NonlinearHistogram ();

    double count (int i) const;

    std::vector<double>::const_iterator begin () const;
    std::vector<double>::const_iterator end () const;
    

    double min_val () const;
    double max_val () const;
    int n_bins () const;

    void fill (const std::vector<double>& values,
               const std::vector<double>& weights);

    void fill_presorted (const std::vector<double>& sorted_values,
                         const std::vector<double>& sorted_weights);


    int index_for_value (double value) const;

    double value_for_index (int i) const;

    void write (std::ostream& os) const;


    static std::vector<std::pair<double,double> > get_sorted_values_weights (
        const std::vector<double> values,
        const std::vector<double> weights);

    static std::pair<std::vector<double>,std::vector<double> >
        get_pair_sorted_values_weights (
            const std::vector<double> values,
            const std::vector<double> weights);

    static std::vector<double> get_ntile_boundaries (
        const int n,
        const std::vector<double> values,
        const std::vector<double> weights);

private:

    typedef std::vector<double> dlist;
    typedef dlist::iterator dlist_iter ;
    typedef dlist::const_iterator dlist_citer;
    typedef std::vector<std::pair<double,double> > pairlist;
    typedef pairlist::const_iterator pairlist_citer;

    static std::vector<double> get_ntile_boundaries_presorted (
        const int n,
        const std::vector<double> sorted_values,
        const std::vector<double> sorted_weights);

    std::vector<double> m_bin_edges;

    double m_underflow;
    double m_overflow;
    std::vector<double> m_bin_values;

};


#endif  /* PYBDT_NONLINEAR_HISTOGRAM_HPP */
