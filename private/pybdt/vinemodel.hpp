// vinemodel.hpp


#ifndef PYBDT_VINEMODEL_HPP
#define PYBDT_VINEMODEL_HPP

#include <string>
#include <vector>
#include <utility>

#include "boost_python.hpp"
#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>

#include "model.hpp"

class VineModel : public Model {
    friend class VineModel_pickle_suite;
public:

    // structors

    VineModel (const std::vector<std::string>& feature_names,
               const std::string& vine_feature,
               const std::vector<double> bin_mins,
               const std::vector<double> bin_maxs,
               const std::vector<boost::shared_ptr<Model> > models);

    virtual ~VineModel ();

    // structors for Python

    VineModel (const boost::python::list& feature_names,
               const std::string& vine_feature,
               const boost::python::list& bin_mins,
               const boost::python::list& bin_maxs,
               const boost::python::list& models);

    // inspectors

    std::vector<double> variable_importance (bool sep_weighted) const;
    boost::python::dict variable_importance_py (bool sep_weighted) const;

protected:

    virtual double base_score (const Scoreable& e, bool use_purity) const;

    std::string m_vine_feature;
    size_t m_vine_feature_i;
    std::vector<double> m_bin_mins;
    std::vector<double> m_bin_maxs;
    std::vector<boost::shared_ptr<Model> > m_models;

};

struct VineModel_pickle_suite : boost::python::pickle_suite {
    static
    boost::python::tuple getinitargs (const VineModel& m);
};


void export_vinemodel ();

#endif  /* PYBDT_VINEMODEL_HPP */
