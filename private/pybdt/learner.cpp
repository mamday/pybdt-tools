// learner.cpp

#include <Python.h>

#include "learner.hpp"

#include "np.hpp"

#include <algorithm>


using namespace std;
using namespace boost;
namespace py = boost::python;


// structors

Learner::Learner (const vector<string>& feature_names,
                  const string& weight_name)
:   m_feature_names (feature_names),
    m_sig_weight_name (weight_name), m_bg_weight_name (weight_name)
{
    init ();
}

Learner::Learner (const vector<string>& feature_names,
                  const string& sig_weight_name,
                  const string& bg_weight_name)
:   m_feature_names (feature_names),
    m_sig_weight_name (sig_weight_name), m_bg_weight_name (bg_weight_name)
{
    init ();
}

Learner::~Learner ()
{
}

// structors for Python

Learner::Learner (const py::list& feature_names,
                  const string& weight_name)
:   m_feature_names (np::list_to_vector<string> (feature_names)),
    m_sig_weight_name (weight_name),
    m_bg_weight_name (weight_name)
{
    init ();
}

Learner::Learner (const py::list& feature_names,
                  const std::string& sig_weight_name,
                  const std::string& bg_weight_name)
:   m_feature_names (np::list_to_vector<string> (feature_names)),
    m_sig_weight_name (sig_weight_name),
    m_bg_weight_name (bg_weight_name)
{
    init ();
}


// structor helper
void
Learner::init ()
{
    sort (m_feature_names.begin (), m_feature_names.end ());
}

// inspectors

std::vector<std::string>
Learner::feature_names () const
{
    return m_feature_names;
}

std::string
Learner::sig_weight_name () const
{
    return m_sig_weight_name;
}

std::string
Learner::bg_weight_name () const
{
    return m_bg_weight_name;
}


// factory methods

boost::shared_ptr<Model>
Learner::train (const DataSet& sig, const DataSet& bg) const
{
    vector<double> sig_weights = m_sig_weight_name.size ()
        ? sig.get_column (m_sig_weight_name)
        : np::ones<double> (sig.n_events ());
    vector<double> bg_weights = m_bg_weight_name.size ()
        ? bg.get_column (m_bg_weight_name)
        : np::ones<double> (bg.n_events ());

    sig_weights = np::div (sig_weights, np::sum (sig_weights));
    bg_weights = np::div (bg_weights, np::sum (bg_weights));

    return train_given_weights (sig, bg, sig_weights, bg_weights);
}

boost::shared_ptr<Model>
Learner::train_given_weights (const DataSet& sig, const DataSet& bg,
                              const vector<double>& sig_weights,
                              const vector<double>& bg_weights) const
{
    // get only columns that this learner uses
    const DataSet& train_sig (DataSet (sig, m_feature_names));
    const DataSet& train_bg (DataSet (bg, m_feature_names));
    // eliminate events with NaN's
    vector<int> sig_keep;
    vector<int> bg_keep;
    const vector<Event>& sig_events (train_sig.events ());
    const vector<Event>& bg_events (train_bg.events ());
    sig_keep.reserve (train_sig.n_events ());
    bg_keep.reserve (train_bg.n_events ());
    typedef vector<Event>::const_iterator citer;
    citer i_ev (sig_events.begin ());
    int i (0);
    for (; i_ev != sig_events.end (); ++i_ev, ++i) {
        if (i_ev->all_finite ()) {
            sig_keep.push_back (i);
        }
    }
    for (i = 0, i_ev = bg_events.begin ();
         i_ev != bg_events.end (); ++i_ev, ++i) {
        if (i_ev->all_finite ()) {
            bg_keep.push_back (i);
        }
    }
    return train_given_everything (
        np::subscript (sig_events, sig_keep),
        np::subscript (bg_events, bg_keep),
        np::subscript (sig_weights, sig_keep),
        np::subscript (bg_weights, bg_keep));
}

void
export_learner ()
{
    using namespace boost::python;

    class_<Learner, boost::noncopyable> (
        "Learner",
        "Train a classification model.", no_init)
        .def ("train", &Learner::train)
        .def ("train_given_weights", &Learner::train_given_weights)
        ;

    register_ptr_to_python <boost::shared_ptr<Learner> > ();
}
