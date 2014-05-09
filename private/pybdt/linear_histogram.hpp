// linear_histogram.hpp


#ifndef PYBDT_LINEAR_HISTOGRAM_HPP
#define PYBDT_LINEAR_HISTOGRAM_HPP

#include <cmath>
#include <vector>

#include "histogram.hpp"

class LinearHistogram : public Histogram {
public:

    LinearHistogram (double min_val, double max_val, int n_bins);
    ~LinearHistogram ();

    double count (int i) const;

    std::vector<double>::const_iterator begin () const;
    std::vector<double>::const_iterator end () const;
    double min_val () const;
    double max_val () const;
    int n_bins () const;

    void fill (double value, double weight=1.0);
    void fill (const std::vector<double>& values);
    void fill (const std::vector<double>& values,
               const std::vector<double>& weights);

    int index_for_value (double value) const;

    double value_for_index (int i) const;


private:

    double m_min_val;
    double m_max_val;
    int m_n_bins;

    double m_bin_width;
    double m_underflow;
    double m_overflow;

    std::vector<double> m_bin_values;

};


#endif  /* PYBDT_LINEAR_HISTOGRAM_HPP */
