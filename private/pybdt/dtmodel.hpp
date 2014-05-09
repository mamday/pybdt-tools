// dtmodel.hpp


#ifndef PYBDT_DTMODEL_HPP
#define PYBDT_DTMODEL_HPP

#include <string>
#include <vector>

#include "boost_python.hpp"
#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>

#include "model.hpp"

class DTModel;

class DTNode : public boost::enable_shared_from_this<DTNode> {
    friend class DTNode_pickle_suite;
    // policy:
    // if left == right = null, it's a leaf.
    // for leaves, feature_id is +1 for signal, -1 for background
public:

    // structors

    // for leaf
    DTNode (double sep_index, double w_sig, double w_bg, int n_sig, int n_bg);

    // for node
    DTNode (double sep_gain, double sep_index,
            int feature_id, double feature_val,
            double w_sig, double w_bg,
            int n_sig, int n_bg,
            boost::shared_ptr<DTNode> left, boost::shared_ptr<DTNode> right);

    virtual ~DTNode ();

    // initialization

    void set_dtmodel (boost::shared_ptr<DTModel> model);
    void set_dtmodel (DTModel* model);

    // inspectors

    int feature_id () const;
    std::string feature_name () const;
    double feature_val () const;
    bool is_leaf () const;
    int max_depth () const;
    double n_bg () const;
    int n_leaves () const;
    int n_total () const;
    double n_sig () const;
    double purity () const;
    double sep_index () const;
    double sep_gain () const;
    int tree_size () const;
    double w_bg () const;
    double w_sig () const;
    double w_total () const;

    boost::shared_ptr<DTNode> left () const;
    boost::shared_ptr<DTNode> right () const;


    // mutators

    void prune ();

    // helpers

    const DTNode& trace (const Scoreable& e);
    std::vector<boost::shared_ptr<DTNode> > trace_full (const Scoreable& e);

    // conversions

    boost::shared_ptr<DTNode> get_copy () const;

protected:

    void calc_aux ();

    double m_sep_gain;
    double m_sep_index;

    int m_feature_id;
    double m_feature_val;

    double m_w_sig;
    double m_w_bg;
    int m_n_sig;
    int m_n_bg;
    double m_purity;

    boost::shared_ptr<DTNode> m_left;
    boost::shared_ptr<DTNode> m_right;

    DTModel* m_dtmodel;

};

struct DTNode_pickle_suite : boost::python::pickle_suite {
    static
    boost::python::tuple getinitargs (const DTNode& n);
};

class DTModel : public Model {
    friend class DTModel_pickle_suite;

public:

    // structors

    DTModel (const std::vector<std::string>& feature_names,
             boost::shared_ptr<DTNode> root);
    // DTModel ();

    virtual ~DTModel ();

    // structors for Python

    DTModel (const boost::python::list& feature_names,
             boost::shared_ptr<DTNode> root);

    // inspectors

    std::vector<double> event_variable_importance (
        const Scoreable& s, bool sep_weighted) const;
    boost::python::dict event_variable_importance_py (
        const boost::python::list& vals, bool sep_weighted) const;
    std::vector<double> variable_importance (bool sep_weighted) const;
    boost::python::dict variable_importance_py (bool sep_weighted) const;

    boost::shared_ptr<DTNode> root ();

protected:

    virtual double base_score (const Scoreable& e, bool use_purity) const;

    boost::shared_ptr<DTNode>  m_root;
};

struct DTModel_pickle_suite : boost::python::pickle_suite {
    static
    boost::python::tuple getinitargs (const DTModel& m);
};


void export_dtmodel ();



inline int
DTNode::feature_id () const
{
    return m_feature_id;
}

inline double
DTNode::feature_val () const
{
    return m_feature_val;
}

inline bool
DTNode::is_leaf () const
{
    return ! bool (m_left);
}

inline double
DTNode::n_bg () const
{
    return m_n_bg;
}

inline double
DTNode::n_sig () const
{
    return m_n_sig;
}

inline double
DTNode::purity () const
{
    return m_purity;
}

inline double
DTNode::sep_index () const
{
    return m_sep_index;
}

inline double
DTNode::sep_gain () const
{
    return m_sep_gain;
}

inline double
DTNode::w_bg () const
{
    return m_w_bg;
}

inline double
DTNode::w_sig () const
{
    return m_w_sig;
}

inline boost::shared_ptr<DTNode>
DTNode::left () const
{
    return m_left;
}

inline boost::shared_ptr<DTNode>
DTNode::right () const
{
    return m_right;
}

inline boost::shared_ptr<DTNode>
DTModel::root ()
{
    return m_root;
}


#endif  /* PYBDT_DTMODEL_HPP */
