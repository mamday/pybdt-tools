// histogram.hpp


#ifndef PYBDT_HISTOGRAM_HPP
#define PYBDT_HISTOGRAM_HPP

#include <cmath>
#include <vector>


class Histogram {
public:

    virtual ~Histogram ();


    virtual double count (int i) const = 0;

    virtual std::vector<double>::const_iterator begin () const = 0;
    virtual std::vector<double>::const_iterator end () const = 0;

    virtual double min_val () const = 0;
    virtual double max_val () const = 0;
    virtual int n_bins () const = 0;

    void fill (const std::vector<double>& values);
    virtual void fill (const std::vector<double>& values,
                       const std::vector<double>& weights) = 0;

    virtual int index_for_value (double value) const = 0;

    virtual double value_for_index (int i) const = 0;

};


#endif  /* PYBDT_HISTOGRAM_HPP */
