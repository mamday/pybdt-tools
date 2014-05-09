// pruner.hpp


#ifndef PYBDT_PRUNER_HPP
#define PYBDT_PRUNER_HPP

#include <boost/shared_ptr.hpp>

#include "dtmodel.hpp"

class Pruner {
public:
    virtual ~Pruner ();

    virtual void prune (boost::shared_ptr<DTModel> tree) = 0;

};


class SameLeafPruner : public Pruner {
public:
    SameLeafPruner ();
    virtual ~SameLeafPruner ();
    virtual void prune (boost::shared_ptr<DTModel> tree);

private:

    void prune_node (boost::shared_ptr<DTNode> node);

};


class CostComplexityPruner : public Pruner {
public:
    explicit CostComplexityPruner (double strength);
    virtual ~CostComplexityPruner ();
    virtual void prune (boost::shared_ptr<DTModel> tree);

    static double gain (const boost::shared_ptr<DTNode> node);
    static double rho (const boost::shared_ptr<DTNode> node);

    double strength () const;
    void strength (double s);

private:
    double m_strength;
};


class ErrorPruner : public Pruner {
public:
    explicit ErrorPruner (double factor);
    virtual ~ErrorPruner ();
    virtual void prune (boost::shared_ptr<DTModel> tree);

    double node_error (const boost::shared_ptr<DTNode> node);
    double subtree_error (const boost::shared_ptr<DTNode> node);
    
    double strength () const;
    void strength (double s);

private:
    std::vector<boost::shared_ptr<DTNode> > get_prune_sequence (
        boost::shared_ptr<DTNode> node);
    double m_strength;
};


void export_pruners ();

#endif  /* PYBDT_PRUNER_HPP */
