// bdtmodel.cpp

#include <Python.h>

#include "bdtmodel.hpp"

#include "np.hpp"

#include <cmath>
#include <boost/make_shared.hpp>

using namespace std;
using namespace boost;
namespace py = boost::python;


BDTModel::BDTModel (const vector<string>& feature_names,
                    const vector<boost::shared_ptr<DTModel> >& dtmodels,
                    const vector<double>& alphas)
: Model (feature_names), m_dtmodels (dtmodels), m_alphas (alphas),
    m_n_dtmodels (dtmodels.size ()), m_max_response (np::sum (alphas))
{
}

BDTModel::~BDTModel ()
{
}

BDTModel::BDTModel (const py::list& feature_names,
                    const py::list& dtmodels,
                    const py::list& alphas)
: Model (feature_names),
    m_dtmodels (np::list_to_vector<boost::shared_ptr<DTModel> > (dtmodels)),
    m_alphas (np::list_to_vector<double> (alphas)),
    m_n_dtmodels (len (dtmodels))
{
    m_max_response = np::sum (m_alphas);
}

double
BDTModel::get_alpha (int n) const
{
    return m_alphas.at (n);
}

boost::shared_ptr<DTModel>
BDTModel::get_dtmodel (int n) const
{
    return m_dtmodels.at (n);
}

int
BDTModel::n_dtmodels () const
{
    return m_n_dtmodels;
}

vector<double>
BDTModel::event_variable_importance (
    const Scoreable& s, bool sep_weighted, bool tree_weighted) const
{
    vector<double> abs_var_imp (m_n_features, 0);
    double abs_var_imp_sum (0);
    for (int m (0); m < m_n_dtmodels; ++m) {
        boost::shared_ptr<DTModel> dtmodel (m_dtmodels[m]);
        if (dtmodel->root ()->max_depth () == 0) {
            continue;
        }
        vector<double> these_var_imp 
            = dtmodel->event_variable_importance (s, sep_weighted);
        const double alpha = m_alphas[m];
        for (int i_f (0); i_f < m_n_features; ++i_f) {
            double this_var_imp = these_var_imp[i_f];
            if (tree_weighted) {
                this_var_imp *= alpha;
            }
            abs_var_imp[i_f] += this_var_imp;
            abs_var_imp_sum += this_var_imp;
        }
    }
    vector<double> rel_var_imp (m_n_features, 0);
    for (int i_f (0); i_f < m_n_features; ++i_f) {
        rel_var_imp[i_f] = abs_var_imp[i_f] / abs_var_imp_sum;
    }
    return rel_var_imp;
}

py::dict
BDTModel::event_variable_importance_py (
    const py::list& vals, bool sep_weighted, bool tree_weighted) const
{
    vector<double> var_imp = event_variable_importance (
        make_scoreable (np::list_to_vector<double> (vals)),
        sep_weighted, tree_weighted);
    py::dict out;
    for (int i_f (0); i_f < m_n_features; ++i_f) {
        out[m_feature_names[i_f]] = var_imp[i_f];
    }
    return out;
}

vector<double>
BDTModel::variable_importance (bool sep_weighted, bool tree_weighted) const
{
    vector<double> abs_var_imp (m_n_features, 0);
    double abs_var_imp_sum (0);
    for (int m (0); m < m_n_dtmodels; ++m) {
        boost::shared_ptr<DTModel> dtmodel (m_dtmodels[m]);
        if (dtmodel->root ()->max_depth () == 0) {
            continue;
        }
        vector<double> these_var_imp 
            = dtmodel->variable_importance (sep_weighted);
        const double alpha = m_alphas[m];
        for (int i_f (0); i_f < m_n_features; ++i_f) {
            double this_var_imp = these_var_imp[i_f];
            if (tree_weighted) {
                this_var_imp *= alpha;
            }
            abs_var_imp[i_f] += this_var_imp;
            abs_var_imp_sum += this_var_imp;
        }
    }
    vector<double> rel_var_imp (m_n_features, 0);
    for (int i_f (0); i_f < m_n_features; ++i_f) {
        rel_var_imp[i_f] = abs_var_imp[i_f] / abs_var_imp_sum;
    }
    return rel_var_imp;
}

py::dict
BDTModel::variable_importance_py (bool sep_weighted, bool tree_weighted) const
{
    vector<double> var_imp = variable_importance (
        sep_weighted, tree_weighted);
    py::dict out;
    for (int i_f (0); i_f < m_n_features; ++i_f) {
        out[m_feature_names[i_f]] = var_imp[i_f];
    }
    return out;
}

boost::shared_ptr<BDTModel>
BDTModel::get_subset_bdtmodel (int n_i, int n_f) const
{
    vector<boost::shared_ptr<DTModel> > subset_dtmodels;
    vector<double> subset_alphas;
    if (n_i < 0 or n_i >= m_n_dtmodels or n_f < 0 or n_f > m_n_dtmodels
        or n_i >= n_f) {
        throw std::runtime_error ("invalid range given for DTModel subset");
    }
    subset_dtmodels.reserve (n_f - n_i);
    subset_alphas.reserve (n_f - n_i);
    for (int i (n_i); i < n_f; ++i) {
        subset_dtmodels.push_back (m_dtmodels.at (i));
        subset_alphas.push_back (m_alphas.at (i));
    }
    return boost::make_shared<BDTModel> (
        m_feature_names, subset_dtmodels, subset_alphas);
}

boost::shared_ptr<BDTModel>
BDTModel::get_subset_bdtmodel_list (py::list dtmodel_indices) const
{
    vector<boost::shared_ptr<DTModel> > subset_dtmodels;
    vector<double> subset_alphas;
    const vector<int> indices (np::list_to_vector<int> (dtmodel_indices));
    for (vector<int>::const_iterator i_index (indices.begin ());
         i_index != indices.end (); ++i_index) {
        int index (*i_index);
        if (index < 0 or index >= m_n_dtmodels) {
            throw std::runtime_error ("invalid index given for DTModel");
        }
        subset_dtmodels.push_back (m_dtmodels.at (index));
        subset_alphas.push_back (m_alphas.at (index));
    }
    return boost::make_shared<BDTModel> (
        m_feature_names, subset_dtmodels, subset_alphas);
}

boost::shared_ptr<BDTModel>
BDTModel::get_trimmed_bdtmodel (double threshold) const
{
    vector<boost::shared_ptr<DTModel> > subset_dtmodels;
    vector<double> subset_alphas;
    subset_dtmodels.reserve (n_dtmodels ());
    subset_alphas.reserve (n_dtmodels ());
    // first pass: find max_param
    double max_param (0.);
    for (int i (1); i < n_dtmodels () ; ++i) {
        double d_alpha (m_alphas[i] - m_alphas[i-1]);
        double param (fabs (d_alpha));
        max_param = max (max_param, param);
    }
    // always include first dtmodel
    subset_dtmodels.push_back (m_dtmodels[0]);
    subset_alphas.push_back (m_alphas[0]);
    for (int i (1); i < n_dtmodels (); ++i) {
        double d_alpha (m_alphas[i] - m_alphas[i-1]);
        double param (fabs (d_alpha));
        if (param / max_param > threshold / 100.) {
            // include any dtmodel sufficiently different from the previous
            subset_dtmodels.push_back (m_dtmodels[i]);
            subset_alphas.push_back (m_alphas[i]);
        }
    }
    return boost::make_shared<BDTModel> (
        m_feature_names, subset_dtmodels, subset_alphas);
}

double
BDTModel::base_score (const Scoreable& e, bool use_purity) const
{
    double score (0.);
    for (int m = 0; m < m_n_dtmodels; ++m) {
        score += m_dtmodels[m]->score (e, use_purity) * m_alphas[m]
            / m_max_response;
    }
    return min (1., max (-1., score));
}

py::tuple
BDTModel_pickle_suite::getinitargs (const BDTModel& m)
{
    return py::make_tuple (
        np::vector_to_list (m.m_feature_names),
        np::vector_to_list (m.m_dtmodels),
        np::vector_to_list (m.m_alphas));
}

void
export_bdtmodel ()
{

    typedef const py::list& clr;

    using namespace boost::python;
    class_<BDTModel, bases<Model> > (
        "BDTModel", init<clr,clr,clr> ())
        .def ("get_alpha", &BDTModel::get_alpha)
        .def ("get_dtmodel", &BDTModel::get_dtmodel)
        .def ("get_subset_bdtmodel", &BDTModel::get_subset_bdtmodel)
        .def ("get_subset_bdtmodel_list", &BDTModel::get_subset_bdtmodel_list)
        .def ("get_trimmed_bdtmodel", &BDTModel::get_trimmed_bdtmodel)
        .def ("event_variable_importance",
              &BDTModel::event_variable_importance_py)
        .def ("variable_importance",
              &BDTModel::variable_importance_py)
        .add_property ("n_dtmodels", &BDTModel::n_dtmodels)
        .def_pickle (BDTModel_pickle_suite ())
        ;

    register_ptr_to_python <boost::shared_ptr<BDTModel> > ();
}
