// dtmodel.cpp

#include <Python.h>

#include "vinemodel.hpp"
#include "np.hpp"

#include <iostream>
#include <limits>

#include <boost/math/special_functions/fpclassify.hpp>

using namespace std;
using namespace boost;
using namespace boost::python;
namespace py = boost::python;


VineModel::VineModel (const vector<string>& feature_names,
                      const string& vine_feature,
                      const vector<double> bin_mins,
                      const vector<double> bin_maxs,
                      const vector<boost::shared_ptr<Model> > models)
:   Model (feature_names),
    m_vine_feature (vine_feature),
    m_bin_mins (bin_mins), m_bin_maxs (bin_maxs), m_models (models)
{
    for (size_t i (0); i < m_feature_names.size (); ++i) {
        if (m_feature_names[i] == m_vine_feature) {
            m_vine_feature_i = i;
            break;
        }
    }
}

VineModel::VineModel (const py::list& feature_names,
                      const string& vine_feature,
                      const py::list& bin_mins,
                      const py::list& bin_maxs,
                      const py::list& models)
:   Model (feature_names),
    m_vine_feature (vine_feature),
    m_bin_mins (np::list_to_vector<double> (bin_mins)),
    m_bin_maxs (np::list_to_vector<double> (bin_maxs)),
    m_models (np::list_to_vector<boost::shared_ptr<Model> > (models))
{
    for (size_t i (0); i < m_feature_names.size (); ++i) {
        if (m_feature_names[i] == m_vine_feature) {
            m_vine_feature_i = i;
            break;
        }
    }
}

VineModel::~VineModel ()
{
}

double
VineModel::base_score (const Scoreable& e, bool use_purity) const
{
    vector<double> scores;
    scores.reserve (8);
    for (size_t m (0); m < m_models.size (); ++m) {
        double bin_min (m_bin_mins[m]);
        double bin_max (m_bin_maxs[m]);
        double value (e[m_vine_feature_i]);
        if ((bin_min <= value) and (value < bin_max)) {
            scores.push_back (m_models[m]->score (e, use_purity));
        }
    }
    return np::sum (scores) / scores.size ();
}


py::tuple
VineModel_pickle_suite::getinitargs (const VineModel& m)
{
    return py::make_tuple (
        np::vector_to_list (m.m_feature_names),
        m.m_vine_feature,
        np::vector_to_list (m.m_bin_mins),
        np::vector_to_list (m.m_bin_maxs),
        np::vector_to_list (m.m_models));
}


void export_vinemodel ()
{
    class_<VineModel, bases<Model> > (
        "VineModel", init<list,string,list,list,list> ())
        .def_pickle (VineModel_pickle_suite ())
        ;
}
