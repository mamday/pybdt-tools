// dtlearner.cpp

#include <Python.h>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include "dtlearner.hpp"

#include "linear_histogram.hpp"
#include "nonlinear_histogram.hpp"
#include "np.hpp"

#include <limits>
#include <map>


using namespace std;
using namespace boost;
namespace py = boost::python;


DTLearner::DTLearner (const vector<string>& feature_names,
                      const string& weight_name)
: Learner (feature_names, weight_name)
{
    set_defaults ();
}

DTLearner::DTLearner (const vector<string>& feature_names,
                      const string& sig_weight_name,
                      const string& bg_weight_name)
: Learner (feature_names, sig_weight_name, bg_weight_name)
{
    set_defaults ();
}

DTLearner::~DTLearner ()
{
}

DTLearner::DTLearner (const py::list& feature_names,
                      const string& weight_name)
: Learner (feature_names, weight_name)
{
    set_defaults ();
}

DTLearner::DTLearner (
    const py::list& feature_names,
    const string& sig_weight_name,
    const string& bg_weight_name)
: Learner (feature_names, sig_weight_name, bg_weight_name)
{
    int n_names = len (feature_names);
    m_feature_names.resize (n_names);
    for (int j = 0; j < n_names; ++j) {
        string name = py::extract<string> (feature_names[j]);
        m_feature_names[j] = name;
    }
    set_defaults ();
}


int
DTLearner::max_depth () const
{
    return m_max_depth;
}

int
DTLearner::min_split () const
{
    return m_min_split;
}

int
DTLearner::num_cuts () const
{
    return m_num_cuts;
}

bool
DTLearner::linear_cuts () const
{
    return m_linear_cuts;
}

int
DTLearner::num_random_variables () const
{
    return m_num_random_variables;
}

std::string
DTLearner::separation_type () const
{
    return m_sep_func->separation_type ();
}

void
DTLearner::max_depth (int n)
{
    m_max_depth = n;
}

void
DTLearner::min_split (int n)
{
    m_min_split = n;
}

void
DTLearner::num_cuts (int n)
{
    m_num_cuts = n;
}

void
DTLearner::linear_cuts (bool value)
{
    m_linear_cuts = value;
}

void
DTLearner::num_random_variables (int n)
{
    m_num_random_variables = n;
}

void
DTLearner::separation_type (std::string st)
{
    if (st == "gini") {
        m_sep_func = boost::make_shared<SepGini> ();
    }
    else if (st == "cross_entropy") {
        m_sep_func = boost::make_shared<SepCrossEntropy> ();
    }
    else if (st == "misclass_error") {
        m_sep_func = boost::make_shared<SepMisclassError> ();
    }
    else {
        throw std::runtime_error ("unknown separation type");
    }
}

void
DTLearner::set_defaults ()
{
    separation_type ("gini");
    m_linear_cuts = true;
    m_min_split = 20;
    m_max_depth = 5;
    m_num_cuts = 20;
    m_num_random_variables = 0;
}

RegLearner::RegLearner (const vector<string>& feature_names,
                      const string& sig_weight_name,
                      const string& bg_weight_name)
: DTLearner (feature_names, sig_weight_name, bg_weight_name)
{
    separation_type ("cross_entropy");
}

RegLearner::RegLearner (
    const py::list& feature_names,
    const string& sig_weight_name,
    const string& bg_weight_name)
: DTLearner (feature_names, sig_weight_name, bg_weight_name)
{
    separation_type ("cross_entropy");
}

RegLearner::~RegLearner ()
{
}

boost::shared_ptr<Model>
DTLearner::train_given_everything (const vector<Event>& sig,
                                   const vector<Event>& bg,
                                   const vector<double>& sig_weights,
                                   const vector<double>& bg_weights) const
{
    assert (sig.size () == sig_weights.size ());
    assert (bg.size () == bg_weights.size ());
    boost::shared_ptr<DTNode> root = build_tree (
        sig, bg, sig_weights, bg_weights);
    return boost::make_shared<DTModel> (m_feature_names, root);
}

boost::shared_ptr<DTNode>
DTLearner::build_tree (
    const vector<Event>& sig_events,
    const vector<Event>& bg_events,
    const vector<double>& sig_weights, const vector<double>& bg_weights,
    const int depth) const
{
    using namespace np;

    typedef vector<Event> vecev;
    typedef vector<double> vecd;
    typedef vecev::const_iterator vecev_citer;
    typedef vecd::const_iterator vecd_citer;

    int n_sig (sig_events.size ());
    int n_bg (bg_events.size ());
    double w_sig (np::sum (sig_weights));
    double w_bg (np::sum (bg_weights));
    const double w_here (w_sig + w_bg);
    const double purity_here (w_sig / w_here);
    const double sep_here ((*m_sep_func) (purity_here));

    // if too few events, max depth, or all one type, then make leaf now
    if ((n_sig + n_bg < m_min_split)
        or depth == m_max_depth or n_sig == 0 or n_bg == 0) {
        return boost::make_shared<DTNode> (sep_here, w_sig, w_bg, n_sig, n_bg);
    }

    const int n_available_features (m_feature_names.size ());
    const int n_split_features (
        m_num_random_variables ? m_num_random_variables : n_available_features);

    // get indices of features in m_feature_names
    vector<int> split_features_i;
    if (n_split_features == n_available_features) {
        split_features_i = np::range<int> (n_available_features);
    }
    else {
        split_features_i = m_random_sampler.sample_range<int> (
            n_split_features, 0, n_available_features);
    }

    // below, always iterate over i_i_f in [0, n_split_features)
    //
    // n_split_features is the size of the vectors of feature limits
    // and vectors of histograms
    //
    // to query the events, i_f = split_features_i[i_i_f]
    // and feature_name = m_feature_names[i_f]


    vecev_citer i_ev;
    vecd_citer i_weight;

    // get extreme values of features
    vector<double> feature_mins (n_split_features);
    vector<double> feature_maxs (n_split_features);
    for (i_ev = sig_events.begin ();
         i_ev != sig_events.end (); ++i_ev) {
        for (int i_i_f (0); i_i_f < n_split_features; ++i_i_f) {
            const int i_f (split_features_i[i_i_f]);
            double& feature_min = feature_mins[i_i_f];
            double& feature_max = feature_maxs[i_i_f];
            const double feature_val = i_ev->value (i_f);
            if (i_ev == sig_events.begin () or feature_val < feature_min) {
                feature_min = feature_val;
            }
            if (i_ev == sig_events.begin () or feature_val > feature_max) {
                feature_max = feature_val;
            }
        }
    }
    for (i_ev = bg_events.begin ();
         i_ev != bg_events.end (); ++i_ev) {
        for (int i_i_f (0); i_i_f < n_split_features; ++i_i_f) {
            const int i_f (split_features_i[i_i_f]);
            double& feature_min = feature_mins[i_i_f];
            double& feature_max = feature_maxs[i_i_f];
            const double feature_val = i_ev->value (i_f);
            if (feature_val < feature_min) {
                feature_min = feature_val;
            }
            if (feature_val > feature_max) {
                feature_max = feature_val;
            }
        }
    }

    // vecd all_weights (sig_weights);
    // all_weights.insert (
    //     all_weights.end (), bg_weights.begin (), bg_weights.end ());

    // create sig and bg, weighted and unweighted histogram for each feature
    vector<boost::shared_ptr<Histogram> > w_sig_hists;
    vector<boost::shared_ptr<Histogram> > w_bg_hists;
    vector<boost::shared_ptr<Histogram> > n_sig_hists;
    vector<boost::shared_ptr<Histogram> > n_bg_hists;
    for (int i_i_f (0); i_i_f < n_split_features; ++i_i_f) {
        // get values and weights lists
        //cerr << "variable " << i_i_f << endl;
        const int i_f (split_features_i[i_i_f]);
        vecd sig_values;
        for (i_ev = sig_events.begin (); i_ev != sig_events.end (); ++i_ev) {
            const double value = i_ev->value (i_f);
            sig_values.push_back (value);
        }
        vecd bg_values;
        for (i_ev = bg_events.begin (); i_ev != bg_events.end (); ++i_ev) {
            const double value = i_ev->value (i_f);
            bg_values.push_back (value);
        }
        boost::shared_ptr<NonlinearHistogram> h;
        if (m_linear_cuts) {
	  w_sig_hists.push_back (boost::make_shared<LinearHistogram> (
                    feature_mins[i_i_f], feature_maxs[i_i_f], m_num_cuts + 1));
	  w_bg_hists.push_back (boost::make_shared<LinearHistogram> (
                    feature_mins[i_i_f], feature_maxs[i_i_f], m_num_cuts + 1));
	  n_sig_hists.push_back (boost::make_shared<LinearHistogram> (
                    feature_mins[i_i_f], feature_maxs[i_i_f], m_num_cuts + 1));
	  n_bg_hists.push_back (boost::make_shared<LinearHistogram> (
                    feature_mins[i_i_f], feature_maxs[i_i_f], m_num_cuts + 1));
            w_sig_hists[i_i_f]->fill (sig_values, sig_weights);
            n_sig_hists[i_i_f]->fill (sig_values);
            w_bg_hists[i_i_f]->fill (bg_values, bg_weights);
            n_bg_hists[i_i_f]->fill (bg_values);
        }
        else {
            pair<vecd,vecd> sig_sorted_values_weights_pair (
                NonlinearHistogram::get_pair_sorted_values_weights (
                    sig_values, sig_weights));
            vecd sig_sorted_values (sig_sorted_values_weights_pair.first);
            vecd sig_sorted_weights (sig_sorted_values_weights_pair.second);

            pair<vecd,vecd> bg_sorted_values_weights_pair (
                NonlinearHistogram::get_pair_sorted_values_weights (
                    bg_values, bg_weights));
            vecd bg_sorted_values (bg_sorted_values_weights_pair.first);
            vecd bg_sorted_weights (bg_sorted_values_weights_pair.second);

            vecd all_values (sig_sorted_values);
            all_values.insert (
                all_values.end (),
                bg_sorted_values.begin (), bg_sorted_values.end ());

            vecd all_sorted_weights (sig_sorted_weights);
            all_sorted_weights.insert (
                all_sorted_weights.end (),
                bg_sorted_weights.begin (), bg_sorted_weights.end ());

            vecd bin_edges (NonlinearHistogram::get_ntile_boundaries (
                    m_num_cuts, all_values, all_sorted_weights));

            h = boost::make_shared<NonlinearHistogram> (bin_edges);
            h->fill_presorted (sig_sorted_values, sig_sorted_weights);
            w_sig_hists.push_back (h);

            h = boost::make_shared<NonlinearHistogram> (bin_edges);
            h->fill_presorted (bg_sorted_values, bg_sorted_weights);
            w_bg_hists.push_back (h);

            h = boost::make_shared<NonlinearHistogram> (bin_edges);
            h->fill_presorted (
                sig_sorted_values,
                np::ones<double> (sig_sorted_values.size ()));
            n_sig_hists.push_back (h);

            h = boost::make_shared<NonlinearHistogram> (bin_edges);
            h->fill_presorted (
                bg_sorted_values,
                np::ones<double> (bg_sorted_values.size ()));
            n_bg_hists.push_back (h);
        }
    }

    // scan each pair (sig,bg) of histograms for best separation gain
    double best_sep_gain (-1);
    double best_sep_index (-1);
    int best_i_f (-1);
    double best_cut_val (numeric_limits<double>::quiet_NaN ());
    for (int i_i_f (0); i_i_f < n_split_features; ++i_i_f) {
        int i_f (split_features_i[i_i_f]);
        const Histogram& w_h_sig = *w_sig_hists[i_i_f];
        const Histogram& w_h_bg = *w_bg_hists[i_i_f];
        const Histogram& n_h_sig = *n_sig_hists[i_i_f];
        const Histogram& n_h_bg = *n_bg_hists[i_i_f];
        double w_sig_left (0), w_sig_right (w_sig);
        double w_bg_left (0), w_bg_right (w_bg);
        double n_sig_left (0), n_sig_right (n_sig);
        double n_bg_left (0), n_bg_right (n_bg);
        // cut is at right edge of bin -- don't bother checking last bin
        vecd_citer i_w_h_sig (w_h_sig.begin ());
        vecd_citer i_n_h_sig (n_h_sig.begin ());
        vecd_citer i_w_h_bg (w_h_bg.begin ());
        vecd_citer i_n_h_bg (n_h_bg.begin ());
        int i_bin (0);
        const int n_bins (w_h_sig.n_bins ());
        for (; i_bin < n_bins - 1;
             ++i_bin, ++i_w_h_sig, ++i_n_h_sig, ++i_w_h_bg, ++i_n_h_bg) {
             w_sig_left += *i_w_h_sig;
             w_bg_left += *i_w_h_bg;
             n_sig_left += *i_n_h_sig;
             n_bg_left += *i_n_h_bg;
             w_sig_right -= *i_w_h_sig;
             w_bg_right -= *i_w_h_bg;
             n_sig_right -= *i_n_h_sig;
             n_bg_right -= *i_n_h_bg;
            const double n_left = n_sig_left + n_bg_left;
            const double n_right = n_sig_right + n_bg_right;
            if (n_left < m_min_split) {
                continue; // not enough to the left yet
            }
            if (n_right < m_min_split) {
                break; // not enough remaining to the right anymore
            }
            const double w_left = w_sig_left + w_bg_left;
            const double w_right = w_sig_right + w_bg_right;
            if (w_left <= 0) {
                // cerr << endl << "w_left = " << w_left
                //     << ", yet n_left = " << n_left;
                // cerr << endl << "w_sig_left = " << w_sig_left
                //     << " and w_bg_left = " << w_bg_left;
                // cerr << endl << "w_sig_right = " << w_sig_right
                //     << " and w_bg_right = " << w_bg_right;
            }
            if (w_right <= 0) {
                // cerr << endl << "w_right = " << w_right
                //     << ", yet n_right = " << n_right;
                // cerr << endl << "w_sig_left = " << w_sig_left
                //     << " and w_bg_left = " << w_bg_left;
                // cerr << endl << "w_sig_right = " << w_sig_right
                //     << " and w_bg_right = " << w_bg_right;
            }
            const double purity_left = w_sig_left / w_left;
            const double purity_right = w_sig_right / w_right;
            const double sep_left = (*m_sep_func) (purity_left);
            const double sep_right = (*m_sep_func) (purity_right);
            const double sep_gain = w_here * sep_here
                - (w_left * sep_left) - (w_right * sep_right);
            if (sep_gain > best_sep_gain) {
                best_sep_gain = sep_gain;
                best_sep_index = sep_here;
                best_i_f = i_f;
                best_cut_val = w_h_sig.value_for_index (i_bin + 1);
            }
        }
    }

    // use chosen split
    if (best_i_f >= 0) {
        //cerr << setw (depth) << ' '
        //    << "cutting on " << m_feature_names[best_i_f]
        //    << " at " << best_cut_val
        //    << endl;
        // calculate left and right event subsets
        vecev sig_left;
        vecev sig_right;
        vecev bg_left;
        vecev bg_right;
        sig_left.reserve (n_sig);
        sig_right.reserve (n_sig);
        bg_left.reserve (n_bg);
        bg_right.reserve (n_bg);
        vector<double> sig_weights_left;
        vector<double> sig_weights_right;
        vector<double> bg_weights_left;
        vector<double> bg_weights_right;
        sig_weights_left.reserve (n_sig);
        sig_weights_right.reserve (n_sig);
        bg_weights_left.reserve (n_bg);
        bg_weights_right.reserve (n_bg);

        for (i_ev = sig_events.begin (), i_weight = sig_weights.begin ();
             i_ev != sig_events.end (); ++i_ev, ++i_weight) {
            const Event& event = *i_ev;
            double weight = *i_weight;
            if (event.value (best_i_f) < best_cut_val) {
                sig_left.push_back (event);
                sig_weights_left.push_back (weight);
            }
            else {
                sig_right.push_back (event);
                sig_weights_right.push_back (weight);
            }
        }
        for (i_ev = bg_events.begin (), i_weight = bg_weights.begin ();
             i_ev != bg_events.end (); ++i_ev, ++i_weight) {
            const Event& event = *i_ev;
            double weight = *i_weight;
            if (event.value (best_i_f) < best_cut_val) {
                bg_left.push_back (event);
                bg_weights_left.push_back (weight);
            }
            else {
                bg_right.push_back (event);
                bg_weights_right.push_back (weight);
            }
        }

        boost::shared_ptr<DTNode> left (build_tree (
                sig_left, bg_left,
                sig_weights_left, bg_weights_left,
                depth + 1));
        boost::shared_ptr<DTNode> right (build_tree (
                sig_right, bg_right,
                sig_weights_right, bg_weights_right,
                depth + 1));
        assert (left);
        assert (right);
        return boost::shared_ptr<DTNode> (new DTNode (
                best_sep_gain, best_sep_index, best_i_f, best_cut_val,
                w_sig, w_bg, n_sig, n_bg, left, right));
    }
    return boost::make_shared<DTNode> (sep_here, w_sig, w_bg, n_sig, n_bg);
}

void
export_dtlearner ()
{
    using namespace boost::python;
   
    class_<DTLearner, bases<Learner> > (
        "DTLearner",
        "Train a single decision tree."
        ,init<py::list>())
        .def (init<py::list,string> ())
        .def (init<py::list,string,string> ())
        .def ("set_defaults", &DTLearner::set_defaults)
        .add_property (
            "linear_cuts", 
            (bool (DTLearner::*)()const) &DTLearner::linear_cuts,
            (void (DTLearner::*)(bool)) &DTLearner::linear_cuts)
        .add_property (
            "max_depth", 
            (int (DTLearner::*)()const) &DTLearner::max_depth,
            (void (DTLearner::*)(int)) &DTLearner::max_depth)
        .add_property (
            "min_split", 
            (int (DTLearner::*)()const) &DTLearner::min_split,
            (void (DTLearner::*)(int)) &DTLearner::min_split)
        .add_property (
            "num_cuts", 
            (int (DTLearner::*)()const) &DTLearner::num_cuts,
            (void (DTLearner::*)(int)) &DTLearner::num_cuts)
        .add_property (
            "num_random_variables",
            (int (DTLearner::*)()const) &DTLearner::num_random_variables,
            (void (DTLearner::*)(int)) &DTLearner::num_random_variables)
        .add_property (
            "separation_type",
            (std::string (DTLearner::*)()const)&DTLearner::separation_type,
            (void (DTLearner::*)(std::string)) &DTLearner::separation_type)
        ;
        class_<RegLearner,bases<DTLearner> >(
        "RegLearner",
        "Train a single regression tree."
        ,init<py::list,string,string>())
        //.def (init<py::list,string> ())
        .def (init<py::list,string,string> ())
        ;
    register_ptr_to_python <boost::shared_ptr<DTLearner> > ();
    register_ptr_to_python <boost::shared_ptr<RegLearner> > ();
}
