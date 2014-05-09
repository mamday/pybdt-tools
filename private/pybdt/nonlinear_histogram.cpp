// histogram.cpp


#include "nonlinear_histogram.hpp"

#include <iostream>

#include "np.hpp"


using namespace std;

template<typename T>
vector<T>
sorted (const vector<T>& v)
{
    vector<T> out (v);
    sort (out.begin (), out.end ());
    return out;
}

NonlinearHistogram::NonlinearHistogram (const vector<double>& bin_edges)
:   m_bin_edges (sorted (bin_edges)),
    m_bin_values (bin_edges.size () - 1)
{
}

NonlinearHistogram::NonlinearHistogram (const int n,
                                        const std::vector<double> values,
                                        bool presorted)
{
    dlist weights = np::ones<double> (values.size ());
    if (presorted) {
        m_bin_edges = get_ntile_boundaries_presorted (
            n, values, weights);
        this->fill_presorted (values, weights);
    }
    else {
        dlist sorted_values (sorted (values));
        m_bin_edges = get_ntile_boundaries_presorted (
            n, sorted_values, weights);
        this->fill_presorted (sorted_values, weights);
    }
}

NonlinearHistogram::NonlinearHistogram (const int n,
                                        const std::vector<double> values,
                                        const std::vector<double> weights,
                                        bool presorted)
{
    pair<dlist,dlist> values_weights_pair (get_pair_sorted_values_weights (
            values, weights));
    if (presorted) {
        m_bin_edges = get_ntile_boundaries_presorted (
            n, values, weights);
        this->fill_presorted (values, weights);
    }
    else {
        dlist sorted_values (sorted (values));
        dlist sorted_weights (values_weights_pair.second);
        m_bin_edges = get_ntile_boundaries_presorted (
            n, sorted_values, sorted_weights);
        this->fill_presorted (sorted_values, sorted_weights);
    }
}

NonlinearHistogram::~NonlinearHistogram ()
{
}

vector<pair<double,double> >
NonlinearHistogram::get_sorted_values_weights (
    const vector<double> values, const vector<double> weights)
{
    assert (values.size () == weights.size ());
    pairlist values_weights;
    values_weights.reserve (values.size ());
    for (dlist_citer i_values (values.begin ()),
         i_weights (weights.begin ());
         i_values != values.end ();
         ++i_values, ++i_weights) {
        values_weights.push_back (make_pair (*i_values, *i_weights));
    }
    sort (values_weights.begin (), values_weights.end ());
    return values_weights;
}

pair<vector<double>,vector<double> >
NonlinearHistogram::get_pair_sorted_values_weights (
    const std::vector<double> values,
    const std::vector<double> weights)
{
    size_t n_events (values.size ());
    pairlist values_weights (get_sorted_values_weights (values, weights));
    dlist sorted_values;
    dlist sorted_weights;
    sorted_values.reserve (n_events);
    sorted_weights.reserve (n_events);
    for (pairlist_citer i (values_weights.begin ());
         i != values_weights.end (); ++i) {
        sorted_values.push_back (i->first);
        sorted_weights.push_back (i->second);
    }
    return make_pair (sorted_values, sorted_weights);
}


vector<double>
NonlinearHistogram::get_ntile_boundaries (
    const int n, const vector<double> values, const vector<double> weights)
{
    pair<dlist,dlist> values_weights_pair (get_pair_sorted_values_weights (
            values, weights));
    dlist sorted_values (values_weights_pair.first);
    dlist sorted_weights (values_weights_pair.second);
    return get_ntile_boundaries_presorted (
        n, sorted_values, sorted_weights);
}

double
NonlinearHistogram::count (int i) const
{
    return m_bin_values.at (i);
}

vector<double>::const_iterator
NonlinearHistogram::begin () const
{
    return m_bin_values.begin ();
}
vector<double>::const_iterator
NonlinearHistogram::end () const
{
    return m_bin_values.end ();
}

double
NonlinearHistogram::min_val () const {
    return m_bin_edges.front ();
}

double
NonlinearHistogram::max_val () const {
    return m_bin_edges.back ();
}

int
NonlinearHistogram::n_bins () const {
    return m_bin_values.size ();
}

void
NonlinearHistogram::fill (const std::vector<double>& values,
                          const std::vector<double>& weights)
{
    pair<dlist,dlist> values_weights_pair (get_pair_sorted_values_weights (
            values, weights));
    dlist sorted_values (values_weights_pair.first);
    dlist sorted_weights (values_weights_pair.second);
    this->fill_presorted (sorted_values, sorted_weights);
}

int
NonlinearHistogram::index_for_value (double value) const
{
    using namespace std;
    typedef vector<double>::const_iterator dlist_citer;
    if (m_bin_edges.front () <= value and value < m_bin_edges.back ()) {
        int out (0);
        for (dlist_citer i_bin_left (m_bin_edges.begin ()),
             i_bin_right (m_bin_edges.begin () + 1);
             i_bin_right != m_bin_edges.end ();
             ++i_bin_left, ++i_bin_right, ++out) {
            if (*i_bin_left <= value and value < *i_bin_right) {
                return out;
            }
        }
        // control should never reach here
        return m_bin_edges.back ();
    }
    else {
        return -1;
    }
}

double
NonlinearHistogram::value_for_index (int i) const
{
    return m_bin_edges.at (i);
}

void
NonlinearHistogram::write (ostream& os) const
{
    for (int i (0); i < n_bins (); ++i) {
        os << m_bin_edges.at (i) << " to " << m_bin_edges.at (i+1)
            << ": " << m_bin_values.at (i) << endl;
    }
}

void
NonlinearHistogram::fill_presorted (const dlist& sorted_values,
                                    const dlist& sorted_weights)
{
    using namespace std;
    dlist_citer i_bin_left (m_bin_edges.begin ());
    dlist_citer i_bin_right (m_bin_edges.begin () + 1);
    m_bin_values.resize (m_bin_edges.size () - 1, 0);
    dlist_iter i_bin_values (m_bin_values.begin ());
    int int_bin_values (0);
    for (dlist_citer i_values (sorted_values.begin ()),
         i_weights (sorted_weights.begin ());
         i_values != sorted_values.end (); ++i_values, ++i_weights) {
        const double value (*i_values);
        const double weight (*i_weights);
        //cerr << "value " << value
        //    << " vs " << *i_bin_left
        //    << " to " << *i_bin_right << endl;
        if (value < m_bin_edges.front ()) {
            m_underflow += weight;
            //cerr << " --- underflow "
            //    << value << " vs " << m_bin_edges.front () << endl;
            continue;
        }
        else if (value >= m_bin_edges.back ()) {
            m_overflow += weight;
            // cerr << " --- overflow" << endl;
            continue;
        }
        while (value >= *i_bin_right) {
            ++i_bin_left;
            ++i_bin_right;
            ++i_bin_values;
            ++int_bin_values;
        }
        *i_bin_values += weight;
    }
}


vector<double>
NonlinearHistogram::get_ntile_boundaries_presorted (
    const int n,
    const vector<double> sorted_values,
    const vector<double> sorted_weights)
{
    assert (sorted_values.size () == sorted_weights.size ());
    const double total_weight (np::sum (sorted_weights));
    const double ntile_weight (total_weight / n);
    vector<double> bin_edges;
    bin_edges.push_back (sorted_values.front ());
    double weight_so_far (0);
    double total_weight_so_far (0);
    for (dlist_citer i_values (sorted_values.begin ()),
         i_weights (sorted_weights.begin ());
         i_values != sorted_values.end ();
         ++i_values, ++i_weights) {
        const double value (*i_values);
        const double weight (*i_weights);
        weight_so_far += weight;
        total_weight_so_far += weight;
        if (weight_so_far > ntile_weight) {
            bin_edges.push_back (value);
            weight_so_far = weight_so_far + weight - ntile_weight;
        }
        if (int (bin_edges.size ()) == n + 1) {
            break;
        }
    }
    if (int (bin_edges.size ()) < n + 1) {
        bin_edges.push_back (sorted_values.back ());
        
    }
    const double weight = sorted_weights.back ();
    weight_so_far += weight;
    total_weight_so_far += weight;
    // cerr << bin_edges.size () << " of " << n + 1 << endl;
    // cerr << weight_so_far << " of " << ntile_weight
    //     << "; " << total_weight_so_far << " of "<< total_weight << endl;
    return bin_edges;
}

