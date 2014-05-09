// model.cpp

#include <Python.h>

#include "model.hpp"

#include <cmath>
#include <limits>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "np.hpp"
#include "notifier.hpp"


using namespace std;
using namespace boost;
using namespace boost::python;

namespace py = boost::python;



// Scoreable --------------------------------------------------------


Scoreable::~Scoreable ()
{
}

inline bool
Scoreable::all_finite () const
{
    size_type n (size ());
    const Scoreable& self (*this);
    for (size_type j_col (0); j_col < n; ++j_col) {
        if (not std::isfinite (self[j_col])) {
            return false;
        }
    }
    return true;
}


// Model ------------------------------------------------------------

Model::Model (vector<string> feature_names)
    : m_feature_names (feature_names), m_n_features (feature_names.size ())
{
}

Model::Model (boost::python::list feature_names)
    : m_feature_names (np::list_to_vector<string> (feature_names)),
    m_n_features (len (feature_names))
{
}

Model::~Model ()
{
}

vector<string>
Model::feature_names () const
{
    return m_feature_names;
}

py::list
Model::feature_names_py () const
{
    return np::vector_to_list (m_feature_names);
}

double
Model::score (const Scoreable& s, bool use_purity)
{
    if (s.all_finite ()) {
        return base_score (s, use_purity);
    }
    else {
        return std::numeric_limits<double>::quiet_NaN ();
    }
}

vector<double>
Model::score (const DataSet& ds, bool use_purity, bool quiet)
{
    return score (DataSet (ds, m_feature_names).events (), use_purity, quiet);
}

vector<double>
Model::score (const vector<Event>& events, bool use_purity, bool quiet)
{
    typedef vector<Event> vecev;
    typedef vector<double> vecd;
    typedef vecev::const_iterator vecev_citer;
    typedef vecd::const_iterator vecd_citer;
    typedef vecd::iterator vecd_iter;
    int n_events = events.size ();
    vecd scores (n_events);
    vecev_citer i_ev (events.begin ());
    vecd_iter i_score (scores.begin ());
    Notifier<int> notifier ("scoring events", n_events);
    if (not quiet) {
        notifier.update (0);
    }
    for (int count (0); i_ev != events.end (); ++count, ++i_ev, ++i_score) {
        *i_score = score (make_scoreable (*i_ev), use_purity);
        if ((count + 1) % 5000 == 0 and not quiet) {
            notifier.update (count + 1);
        }
    }
    if (not quiet) {
        notifier.finish ();
    }
    return scores;
}

double
Model::score_one (const py::list& vals, bool use_purity)
{
    return score (make_scoreable (np::list_to_vector<double> (vals)),
                  use_purity);
}

PyObject*
Model::score_DataSet (const DataSet& ds, bool use_purity, bool quiet)
{
    return np::vector_to_array (score (ds, use_purity, quiet));
}

void
export_model ()
{
    class_<Model, boost::noncopyable> (
        "Model",
        no_init)
        .def ("score_DataSet", &Model::score_DataSet)
        .def ("score_event", &Model::score_one)
        .add_property ("feature_names", &Model::feature_names_py)
        ;

    register_ptr_to_python <boost::shared_ptr<Model> > ();
}
