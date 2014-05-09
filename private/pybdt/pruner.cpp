// pruner.cpp

#include <Python.h>

#include "pruner.hpp"

#include <map>
#include <numeric>
#include <queue>


using namespace std;
using namespace boost;
namespace py = boost::python;


// Pruner -----------------------------------------------------------

Pruner::~Pruner ()
{
}


// SameLeafPruner ---------------------------------------------------

SameLeafPruner::SameLeafPruner ()
{
}

SameLeafPruner::~SameLeafPruner ()
{
}

void
SameLeafPruner::prune (boost::shared_ptr<DTModel> tree)
{
    prune_node (tree->root ());
}


void
SameLeafPruner::prune_node (boost::shared_ptr<DTNode> node)
{
    if (node->is_leaf ()) {
        return;
    }
    prune_node (node->left ());
    prune_node (node->right ());
    if (node->left ()->is_leaf () and node->right ()->is_leaf ()
        and node->left ()->feature_id () == node->right ()->feature_id ()) {
        node->prune ();
    }
}


// CostComplexityPruner ---------------------------------------------


CostComplexityPruner::CostComplexityPruner (double strength)
    : m_strength (strength)
{
}

CostComplexityPruner::~CostComplexityPruner ()
{
}

void
CostComplexityPruner::prune (boost::shared_ptr<DTModel> tree)
{
    // cerr << endl << "starting ccpruning...";
    int orig_tree_size (tree->root ()->tree_size ());
    typedef boost::shared_ptr<DTNode> pnode;
    pnode test_root (tree->root ()->get_copy ());
    assert (test_root);
    vector<pnode> prune_sequence;
    prune_sequence.reserve (orig_tree_size);
    map<pnode,pnode> real_nodes_by_test_nodes;
    queue<pnode> real_q;
    queue<pnode> test_q;
    // get mapping from test nodes to real nodes
    real_q.push (tree->root ());
    test_q.push (test_root);
    while (!test_q.empty ()) {
        pnode real_node (real_q.front ());
        pnode test_node (test_q.front ());
        assert (real_node);
        assert (test_node);
        real_q.pop ();
        test_q.pop ();
        if (test_node->is_leaf ()) {
            continue;
        }
        real_nodes_by_test_nodes[test_node] = real_node;
        assert (real_nodes_by_test_nodes[test_node]);
        real_q.push (real_node->left ());
        real_q.push (real_node->right ());
        test_q.push (test_node->left ());
        test_q.push (test_node->right ());
    }
    assert (real_nodes_by_test_nodes.find (pnode())
            == real_nodes_by_test_nodes.end ());
    pnode next_test_node_to_prune;
    while ((next_test_node_to_prune != test_root)
           and (not test_root->is_leaf ())) {
        // get next node to prune, till it's the root
        assert (test_root);
        double next_prune_rho (rho (test_root));
        test_q.push (test_root);
        while (!test_q.empty ()) {
            pnode test_node (test_q.front ());
            assert (test_node);
            test_q.pop ();
            if (test_node->is_leaf ()) {
                continue;
            }
            double test_node_rho (rho (test_node));
            if (test_node_rho <= next_prune_rho) {
                // prune node with smallest rho
                // cerr << " .";
                next_test_node_to_prune = test_node;
                next_prune_rho = test_node_rho;
            }
            else {
                if (test_node == test_root) {
                    cerr << "rho didn't match!" << endl;
                    cerr << "-> " << next_prune_rho
                        << " vs. " << test_node_rho << endl;
                    assert (false);
                }
            }
            test_q.push (test_node->left ());
            test_q.push (test_node->right ());
        }
        // prune test tree and save real tree prune sequence
        assert (next_test_node_to_prune);
        pnode next_real_node_to_prune (
            real_nodes_by_test_nodes[next_test_node_to_prune]);
        assert (next_real_node_to_prune);
        prune_sequence.push_back (next_real_node_to_prune);
        next_test_node_to_prune->prune ();
    }
    int prune_count (
       static_cast<int> (m_strength / 100.0 * prune_sequence.size ()));
    for (int i_prune (0); i_prune < prune_count; ++i_prune) {
        assert (prune_sequence.at (i_prune));
        prune_sequence.at (i_prune)->prune ();
    }
    // cerr << endl;
}

double
CostComplexityPruner::strength () const
{
    return m_strength;
}

void
CostComplexityPruner::strength (double s)
{
    m_strength = s;
}

double
CostComplexityPruner::gain (boost::shared_ptr<DTNode> node)
{
    assert (isfinite (node->w_total ()));
    assert (isfinite (node->purity ()));
    return node->w_total () * node->purity () * (1 - node->purity ());
}

double
CostComplexityPruner::rho (boost::shared_ptr<DTNode> node)
{
    if (node->is_leaf ()) {
        return numeric_limits<double>::infinity ();
    }
    else {
        const double c (gain (node));
        const double c_left (gain (node->left ()));
        const double c_right (gain (node->right ()));
        const double rho ((c - (c_left + c_right)) / (node->n_leaves () - 1));
        if (not isfinite (rho)) {
            // cerr << "rho is not finite: " << rho << endl;
        }
        if (isnan (rho) or isnan (-rho)) {
            // cerr << "rho is nan." << endl;
            // cerr << "c: " << c << endl;
            // cerr << "c_left: " << c_left << endl;
            // cerr << "c_right: " << c_right << endl;
            // cerr << "n_leaves: " << node->n_leaves () << endl;
            assert (false);
        }
        return rho;
    }
}


// ErrorPruner ------------------------------------------------------

ErrorPruner::ErrorPruner (double strength)
    : m_strength (strength)
{
}

ErrorPruner::~ErrorPruner ()
{
}

void
ErrorPruner::prune (boost::shared_ptr<DTModel> tree)
{
    typedef vector<boost::shared_ptr<DTNode> > nodelist;
    nodelist prune_sequence (get_prune_sequence (tree->root ()));
    for (nodelist::size_type i_prune (0);
         i_prune < prune_sequence.size (); ++i_prune) {
        prune_sequence[i_prune]->prune ();
    }
}

vector<boost::shared_ptr<DTNode> >
ErrorPruner::get_prune_sequence (boost::shared_ptr<DTNode> node)
{
    typedef vector<boost::shared_ptr<DTNode> > nodelist;
    nodelist out;
    out.reserve (node->tree_size ());
    if (not node->is_leaf ()) {
        vector<boost::shared_ptr<DTNode> > left_seq (
            get_prune_sequence (node->left ()));
        vector<boost::shared_ptr<DTNode> > right_seq (
            get_prune_sequence (node->right ()));
        out.insert (out.end (), left_seq.begin (), left_seq.end ());
        out.insert (out.end (), right_seq.begin (), right_seq.end ());
        if (subtree_error (node) >= node_error (node)) {
            out.push_back (node);
        }
    }
    return out;
}

double
ErrorPruner::node_error (const boost::shared_ptr<DTNode> node)
{
    double w_total (node->w_total ());
    double f (max (node->purity (), 1 - node->purity ()));
    double df (sqrt (f * (1-f) / w_total));
    double error (min (1.0, (1.0 - (f - m_strength * df))));
    return error;
}

double
ErrorPruner::subtree_error (const boost::shared_ptr<DTNode> node)
{
    if (node->is_leaf ()) {
        return node_error (node);
    }
    else {
        boost::shared_ptr<DTNode> left = node->left ();
        boost::shared_ptr<DTNode> right = node->right ();
        return (left->w_total () * subtree_error (left)
                + right->w_total () * subtree_error (right))
            / node->w_total ();
    }
}

double
ErrorPruner::strength () const
{
    return m_strength;
}

void
ErrorPruner::strength (double s)
{
    m_strength = s;
}


void
export_pruners ()
{
    using namespace boost::python;
    class_<Pruner, boost::noncopyable> (
        "Pruner",
        no_init)
        .def ("prune", &Pruner::prune)
        ;

    class_<SameLeafPruner, bases<Pruner> > (
        "SameLeafPruner",
        init<> ())
        ;

    class_<CostComplexityPruner, bases<Pruner> > (
        "CostComplexityPruner",
        init<double> ())
        .add_property (
            "strength", 
            (double (CostComplexityPruner::*)()const)
            &CostComplexityPruner::strength,
            (void (CostComplexityPruner::*)(double))
            &CostComplexityPruner::strength)
        .def ("gain", &CostComplexityPruner::gain).staticmethod ("gain")
        .def ("rho", &CostComplexityPruner::rho).staticmethod ("rho")
        ;

    class_<ErrorPruner, bases<Pruner> > (
        "ErrorPruner",
        init<double> ())
        .add_property (
            "strength", 
            (double (ErrorPruner::*)()const) &ErrorPruner::strength,
            (void (ErrorPruner::*)(double)) &ErrorPruner::strength)
        .def ("subtree_error", &ErrorPruner::subtree_error)
        .def ("node_error", &ErrorPruner::node_error)
        ;

    register_ptr_to_python <boost::shared_ptr<Pruner> > ();
    register_ptr_to_python <boost::shared_ptr<SameLeafPruner> > ();
    register_ptr_to_python <boost::shared_ptr<CostComplexityPruner> > ();
    register_ptr_to_python <boost::shared_ptr<ErrorPruner> > ();
}
