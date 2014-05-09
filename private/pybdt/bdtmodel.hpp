// bdtmodel.hpp


#ifndef PYBDT_BDTMODEL_HPP
#define PYBDT_BDTMODEL_HPP

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>

#include "boost_python.hpp"

#include "dtmodel.hpp"
#include "model.hpp"


class BDTModel : public Model {
    friend class BDTModel_pickle_suite;
public:

    // structors

    BDTModel (const std::vector<std::string>& feature_names,
              const std::vector<boost::shared_ptr<DTModel> >& dtmodels,
              const std::vector<double>& alphas);

    virtual ~BDTModel ();

    // structors for Python

    BDTModel (const boost::python::list& feature_names,
              const boost::python::list& dtmodels,
              const boost::python::list& alphas);


    // inspectors

    double get_alpha (int n) const;
    boost::shared_ptr<DTModel> get_dtmodel (int n) const;
    int n_dtmodels () const;

    std::vector<double> event_variable_importance (
        const Scoreable& s, bool sep_weighted, bool tree_weighted) const;

    boost::python::dict event_variable_importance_py (
        const boost::python::list& vals,
        bool sep_weighted, bool tree_weighted) const;

    std::vector<double> variable_importance (
        bool sep_weighted, bool tree_weighted) const;

    boost::python::dict variable_importance_py (
        bool sep_weighted, bool tree_weighted) const;

    
    // helpers

    boost::shared_ptr<BDTModel> get_subset_bdtmodel (int n_i, int n_f) const;
    boost::shared_ptr<BDTModel> get_subset_bdtmodel_list (
        boost::python::list dtmodel_indices) const;
    boost::shared_ptr<BDTModel> get_trimmed_bdtmodel (double threshold) const;


protected:

    virtual double base_score (const Scoreable& e, bool use_purity) const;


private:
    std::vector<boost::shared_ptr<DTModel> > m_dtmodels;
    std::vector<double> m_alphas;
    int m_n_dtmodels;
    double m_max_response;
};

struct BDTModel_pickle_suite : boost::python::pickle_suite {
    static
    boost::python::tuple getinitargs (const BDTModel& m);
};


void export_bdtmodel ();

#endif  /* PYBDT_BDTMODEL_HPP */
