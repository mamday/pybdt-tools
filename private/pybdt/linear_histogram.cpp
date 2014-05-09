// linear_histogram.cpp


#include "linear_histogram.hpp"
#include "np.hpp"

using namespace std;


LinearHistogram::LinearHistogram (double min_val, double max_val, int n_bins)
:   m_min_val (min_val), m_max_val (max_val), m_n_bins (n_bins),
    m_bin_width ((max_val - min_val) / n_bins),
    m_underflow (0), m_overflow (0),
    m_bin_values (n_bins)
{
}

LinearHistogram::~LinearHistogram ()
{
}

double
LinearHistogram::count (int i) const
{
    return m_bin_values.at (i);
}

vector<double>::const_iterator
LinearHistogram::begin () const
{
    return m_bin_values.begin ();
}


vector<double>::const_iterator
LinearHistogram::end () const
{
    return m_bin_values.end ();
}

double
LinearHistogram::min_val () const
{
    return m_min_val;
}

double
LinearHistogram::max_val () const
{
    return m_max_val;
}

int
LinearHistogram::n_bins () const
{
    return m_n_bins;
}

void
LinearHistogram::fill (double value, double weight)
{
    int i (index_for_value (value));
    if (i >= 0) {
        m_bin_values[i] += weight;
    }
    else if (value < m_min_val) {
        m_underflow += weight;
    }
    else if (value >= m_max_val) {
        m_overflow += weight;
    }
}

void
LinearHistogram::fill (const vector<double>& values,
                       const vector<double>& weights)
{
    typedef vector<double>::const_iterator dlist_citer;
    for (dlist_citer i_value (values.begin ()),
         i_weight (weights.begin ());
        i_value != values.end (); ++i_value, ++i_weight) {
        this->fill (*i_value, *i_weight);
    }
}

int
LinearHistogram::index_for_value (double value) const
{
    if (m_min_val <= value and value < m_max_val) {
        return std::min (int ((value - m_min_val) / m_bin_width),
                         m_n_bins - 1);
    }
    else {
        return -1;
    }
}

double
LinearHistogram::value_for_index (int i) const
{
    return m_min_val + i * m_bin_width;  // left edge value
}

