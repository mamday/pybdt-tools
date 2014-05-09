// dtmodel.cpp

#include <Python.h>

#include "dtmodel.hpp"
#include "np.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>

#include <boost/make_shared.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

using namespace std;
using namespace boost;
using namespace boost::python;
namespace py = boost::python;


// DTNode -----------------------------------------------------------

DTNode::DTNode (double sep_index,
                double w_sig, double w_bg, int n_sig, int n_bg)
:   m_sep_gain (0), m_sep_index (sep_index),
    m_feature_id (w_sig > w_bg ? 1 : -1),
    m_feature_val (numeric_limits<double>::quiet_NaN ()),
    m_w_sig (w_sig), m_w_bg (w_bg),
    m_n_sig (n_sig), m_n_bg (n_bg),
    m_left (), m_right ()
{
    calc_aux ();
}

DTNode::DTNode (double sep_gain, double sep_index,
                int feature_id, double feature_val,
                double w_sig, double w_bg,
                int n_sig, int n_bg,
                boost::shared_ptr<DTNode> left, boost::shared_ptr<DTNode> right)
:   m_sep_gain (sep_gain ), m_sep_index (sep_index),
    m_feature_id (feature_id), m_feature_val (feature_val),
    m_w_sig (w_sig), m_w_bg (w_bg),
    m_n_sig (n_sig), m_n_bg (n_bg),
    m_left (left), m_right (right)
{
    calc_aux ();
}

DTNode::~DTNode ()
{
}

void
DTNode::set_dtmodel (DTModel* model)
{
    m_dtmodel = model;
    if (not is_leaf ()) {
        m_left->set_dtmodel (model);
        m_right->set_dtmodel (model);
    }
}

std::string
DTNode::feature_name () const
{
    if (is_leaf ()) {
        return "[leaf]";
    }
    else {
        using namespace std;
        return m_dtmodel->feature_names ().at (m_feature_id);
    }
}

int
DTNode::tree_size () const
{
    if (is_leaf ()) {
        return 1;
    }
    else {
        return 1 + m_left->tree_size () + m_right->tree_size ();
    }
}

int
DTNode::max_depth () const
{
    if (is_leaf ()) {
        return 0;
    }
    else {
        return 1 + max (m_left->max_depth (), m_right->max_depth ());
    }
}

int
DTNode::n_leaves () const
{
    if (is_leaf ()) {
        return 1;
    }
    else {
        return m_left->n_leaves () + m_right->n_leaves ();
    }
}

int
DTNode::n_total () const
{
    return m_n_sig + m_n_bg;
}

double
DTNode::w_total () const
{
    return m_w_sig + m_w_bg;
}

void
DTNode::prune ()
{
    m_feature_id = m_w_sig > m_w_bg ? 1 : -1;
    m_left = boost::shared_ptr<DTNode> ();
    m_right = boost::shared_ptr<DTNode> ();
}

const DTNode&
DTNode::trace (const Scoreable& e)
{
    if (is_leaf ()) {
        return *this;
    }
    else {
        if (e[m_feature_id] < m_feature_val) {
            return m_left->trace (e);
        }
        else {
            return m_right->trace (e);
        }
    }
}

vector<boost::shared_ptr<DTNode> >
DTNode::trace_full (const Scoreable& e)
{
    vector<boost::shared_ptr<DTNode> > out;
    out.push_back (shared_from_this ());
    vector<boost::shared_ptr<DTNode> > others;
    if (is_leaf ()) {
        return out;
    }
    else {
        if (e[m_feature_id] < m_feature_val) {
            others = m_left->trace_full (e);
        }
        else {
            others = m_right->trace_full (e);
        }
    }
    out.insert (out.end (), others.begin (), others.end ());
    return out;
}

boost::shared_ptr<DTNode>
DTNode::get_copy () const
{
    boost::shared_ptr<DTNode> left, right;
    if (not is_leaf ()) {
        left = m_left->get_copy ();
        right = m_right->get_copy ();
    }
    return boost::shared_ptr<DTNode> (
        new DTNode (m_sep_gain, m_sep_index, m_feature_id, m_feature_val,
                    m_w_sig, m_w_bg, m_n_sig, m_n_bg, left, right));
}

void
DTNode::calc_aux ()
{
    m_purity = m_w_sig / w_total ();
}


py::tuple
DTNode_pickle_suite::getinitargs (const DTNode& n)
{
    return py::make_tuple (
        n.m_sep_gain, n.m_sep_index, n.m_feature_id, n.m_feature_val,
        n.m_w_sig, n.m_w_bg,
        n.m_n_sig, n.m_n_bg,
        n.m_left, n.m_right);
}


// DTModel ----------------------------------------------------------


DTModel::DTModel (const vector<string>& feature_names,
                  boost::shared_ptr<DTNode> root)
: Model (feature_names), m_root (root)
{
    root->set_dtmodel (this);
}

DTModel::~DTModel ()
{ }

DTModel::DTModel (const py::list& feature_names, boost::shared_ptr<DTNode> root)
    : Model (feature_names), m_root (root)
{
    root->set_dtmodel (this);
}


vector<double>
DTModel::event_variable_importance (
    const Scoreable& s, bool sep_weighted) const
{
    vector<double> abs_var_imp (m_n_features, 0);
    vector<boost::shared_ptr<DTNode> > nodes (m_root->trace_full (s));
    double abs_var_imp_sum (0);
    for (vector<boost::shared_ptr<DTNode> >::const_iterator i (nodes.begin ());
         i != nodes.end (); ++i) {
        boost::shared_ptr<DTNode> node (*i);
        if (node->is_leaf ()) {
            continue;
        }
        double weight (node->w_sig () + node->w_bg ());
        if (sep_weighted) {
            double sep_gain (node->sep_gain ());
            double var_imp = sep_gain * sep_gain * weight * weight;
            abs_var_imp[node->feature_id ()] += var_imp;
            abs_var_imp_sum += var_imp;
        }
        else {
            abs_var_imp[node->feature_id ()] += 1;
            abs_var_imp_sum += 1;
        }
    }
    vector<double> rel_var_imp (m_n_features);
    for (int i_f (0); i_f < m_n_features; ++i_f) {
        rel_var_imp[i_f] = abs_var_imp[i_f] / abs_var_imp_sum;
    }
    return rel_var_imp;
}

dict
DTModel::event_variable_importance_py (
    const py::list& vals, bool sep_weighted) const
{
    vector<double> var_imp (event_variable_importance (
            make_scoreable (np::list_to_vector<double> (vals)),
            sep_weighted));
    dict out;
    for (int i_f (0); i_f < m_n_features; ++i_f) {
        out[m_feature_names[i_f]] = var_imp[i_f];
    }
    return out;
}

vector<double>
DTModel::variable_importance (bool sep_weighted) const
{
    vector<double> abs_var_imp (m_n_features, 0);
    queue<boost::shared_ptr<DTNode> > q;
    q.push (m_root);
    double abs_var_imp_sum (0);
    while (!q.empty ()) {
        boost::shared_ptr<DTNode> node (q.front ());
        q.pop ();
        if (node->is_leaf ()) {
            continue;
        }
        double weight (node->w_sig () + node->w_bg ());
        if (sep_weighted) {
            double sep_gain (node->sep_gain ());
            double var_imp = sep_gain * sep_gain * weight * weight;
            abs_var_imp[node->feature_id ()] += var_imp;
            abs_var_imp_sum += var_imp;
        }
        else {
            abs_var_imp[node->feature_id ()] += 1;
            abs_var_imp_sum += 1;
        }
        q.push (node->left ());
        q.push (node->right ());
    }
    vector<double> rel_var_imp (m_n_features);
    for (int i_f (0); i_f < m_n_features; ++i_f) {
        rel_var_imp[i_f] = abs_var_imp[i_f] / abs_var_imp_sum;
    }
    return rel_var_imp;
}

dict
DTModel::variable_importance_py (bool sep_weighted) const
{
    vector<double> var_imp = variable_importance (sep_weighted);
    dict out;
    for (int i_f (0); i_f < m_n_features; ++i_f) {
        out[m_feature_names[i_f]] = var_imp[i_f];
    }
    return out;
}

double
DTModel::base_score (const Scoreable& e, bool use_purity) const
{
    const DTNode& leaf (m_root->trace (e));
    if (use_purity) {
        return 2 * leaf.purity () - 1;
    }
    else {
        return leaf.purity () > 0.5 ? +1 : -1;
    }
}

py::tuple
DTModel_pickle_suite::getinitargs (const DTModel& m)
{
    list py_feature_names;
    for (unsigned i = 0; i < m.m_feature_names.size (); ++i) {
        py_feature_names.append (m.m_feature_names[i]);
    }
    return py::make_tuple (
        py_feature_names, m.m_root);
}

void
export_dtmodel ()
{
    class_<DTNode> (
        "DTNode",
        init<double,double,int,double,double,double,int,int,
        boost::shared_ptr<DTNode>,boost::shared_ptr<DTNode> >())
        .add_property ("feature_id", &DTNode::feature_id)
        .add_property ("feature_name", &DTNode::feature_name)
        .add_property ("feature_val", &DTNode::feature_val)
        .add_property ("is_leaf", &DTNode::is_leaf)
        .add_property ("left", &DTNode::left)
        .add_property ("max_depth", &DTNode::max_depth)
        .add_property ("n_bg", &DTNode::n_bg)
        .add_property ("n_leaves", &DTNode::n_leaves)
        .add_property ("n_sig", &DTNode::n_sig)
        .add_property ("n_total", &DTNode::n_total)
        .add_property ("purity", &DTNode::purity)
        .add_property ("right", &DTNode::right)
        .add_property ("sep_gain", &DTNode::sep_gain)
        .add_property ("sep_index", &DTNode::sep_gain)
        .add_property ("tree_size", &DTNode::tree_size)
        .add_property ("w_bg", &DTNode::w_bg)
        .add_property ("w_sig", &DTNode::w_sig)
        .add_property ("w_total", &DTNode::w_total)
        .def ("prune", &DTNode::prune)
        .def_pickle (DTNode_pickle_suite ())
        ;


    class_<DTModel, bases<Model> > (
        "DTModel", init<const py::list&,boost::shared_ptr<DTNode> > ())
        .def ("event_variable_importance",
              &DTModel::event_variable_importance_py)
        .def ("variable_importance",
              &DTModel::variable_importance_py)
        .add_property ("root", &DTModel::root)
        .def_pickle (DTModel_pickle_suite ())
        ;

    register_ptr_to_python <boost::shared_ptr<DTNode> > ();
    register_ptr_to_python <boost::shared_ptr<DTModel> > ();
}
