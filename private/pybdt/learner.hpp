// learner.hpp

#ifndef PYBDT_LEARNER_HPP
#define PYBDT_LEARNER_HPP


#include <vector>
#include <string>

#include "boost_python.hpp"

#include "dataset.hpp"
#include "model.hpp"

class Learner {
public:

    // structors

    explicit Learner (const std::vector<std::string>& feature_names,
                      const std::string& weight_name="");

    Learner (const std::vector<std::string>& feature_names,
             const std::string& sig_weight_name,
             const std::string& bg_weight_name);

    virtual ~Learner ();

    // structors for Python

    explicit Learner (const boost::python::list& feature_names,
                      const std::string& weight_name="");

    Learner (const boost::python::list& feature_names,
             const std::string& sig_weight_name,
             const std::string& bg_weight_name);

    // inspectors
    std::vector<std::string> feature_names () const;
    std::string sig_weight_name () const;
    std::string bg_weight_name () const;

    // factory methods

    boost::shared_ptr<Model> train (
        const DataSet& sig, const DataSet& bg) const;

    boost::shared_ptr<Model> train_given_weights (
        const DataSet& sig, const DataSet& bg,
        const std::vector<double>& sig_weights,
        const std::vector<double>& bg_weights) const;
    
    virtual boost::shared_ptr<Model> train_given_everything (
        const std::vector<Event>& sig, const std::vector<Event>& bg,
        const std::vector<double>& init_sig_weights,
        const std::vector<double>& init_bg_weights) const = 0;


protected:

    // data members

    std::vector<std::string> m_feature_names;
    std::string m_sig_weight_name;
    std::string m_bg_weight_name;

private:

    void init ();

};

void export_learner ();

#endif  /* PYBDT_LEARNER_HPP */
