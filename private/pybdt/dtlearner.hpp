// dtlearner.hpp


#ifndef PYBDT_DTLEARNER_HPP
#define PYBDT_DTLEARNER_HPP

#include <string>
#include <vector>

#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>

#include "boost_python.hpp"

#include "dataset.hpp"
#include "dtmodel.hpp"
#include "learner.hpp"
#include "random_sampler.hpp"


struct SepFunc {
    virtual double operator() (double p) const = 0;
    virtual std::string separation_type () const = 0;
};
struct SepGini : public SepFunc {
    virtual double operator() (double p) const;
    virtual std::string separation_type () const;
};
struct SepCrossEntropy : public SepFunc {
    virtual double operator() (double p) const;
    virtual std::string separation_type () const;
};
struct SepMisclassError : public SepFunc {
    virtual double operator() (double p) const;
    virtual std::string separation_type () const;
};

inline double SepGini::operator() (double p) const
{
    return p * (1 - p);
}
inline std::string SepGini::separation_type () const
{
    return "gini";
}

inline double SepCrossEntropy::operator() (double p) const
{
    using std::log;
    return -p * log (p) - (1-p) * log (1 - p);
}
inline std::string SepCrossEntropy::separation_type () const
{
    return "cross_entropy";
}

inline double SepMisclassError::operator() (double p) const
{
    return 1 - std::max (p, 1-p);
}
inline std::string SepMisclassError::separation_type () const
{
    return "misclass_error";
}


class DTLearner : public Learner {
public:
    friend class BDTLearner;
    friend class BDTLearner_pickle_suite;


    // structors

    explicit DTLearner (const std::vector<std::string>& feature_names,
                        const std::string& weight_name="");

    DTLearner (const std::vector<std::string>& feature_names,
               const std::string& sig_weight_name,
               const std::string& bg_weight_name);

    virtual ~DTLearner ();

    // structors for Python

    explicit DTLearner (const boost::python::list& feature_names,
                        const std::string& weight_name="");

    DTLearner (const boost::python::list& feature_names,
               const std::string& sig_weight_name,
               const std::string& bg_weight_name);


    // inspectors

    bool linear_cuts () const;
    int max_depth () const;
    int min_split () const;
    int num_cuts () const;
    int num_random_variables () const;
    std::string separation_type () const;

    // mutators

    void max_depth (int n);
    void min_split (int n);
    void num_cuts (int n);
    void linear_cuts (bool value);
    void num_random_variables (int n);
    void separation_type (std::string st);
    void set_defaults ();
    virtual boost::shared_ptr<Model> train_given_everything (
        const std::vector<Event>& sig,
        const std::vector<Event>& bg,
        const std::vector<double>& sig_weights,
        const std::vector<double>& bg_weights) const;


protected:

    boost::shared_ptr<DTNode> build_tree (
        const std::vector<Event>& sig,
        const std::vector<Event>& bg,
        const std::vector<double>& sig_weights,
        const std::vector<double>& bg_weights,
        const int depth=0) const;

    static
    boost::tuple<double, double> m_sum_passing_weight (
        const double cut_val, const std::vector<double>& feature_col,
        const std::vector<double>& weights, const std::vector<int>& idx);


    boost::shared_ptr<SepFunc> m_sep_func;
    int m_min_split;
    int m_max_depth;
    int m_num_cuts;
    bool m_linear_cuts;

    int m_num_random_variables;
    mutable RandomSampler m_random_sampler;
};

class RegLearner : public DTLearner{
public:
    RegLearner (const std::vector<std::string>& feature_names,
               const std::string& sig_weight_name,
               const std::string& bg_weight_name);

    RegLearner (const boost::python::list& feature_names,
               const std::string& sig_weight_name,
               const std::string& bg_weight_name);

    virtual ~RegLearner ();
protected:
    boost::shared_ptr<Model> train_given_targets (
        const std::vector<Event>& sig,
        const std::vector<Event>& bg,
        const std::vector<double>& sig_weights,
        const std::vector<double>& bg_weights,
        const std::vector<double>& sig_targets,
        const std::vector<double>& bg_targets) const;

    boost::shared_ptr<DTNode> build_reg_tree (
        const std::vector<Event>& sig,
        const std::vector<Event>& bg,
        const std::vector<double>& sig_weights,
        const std::vector<double>& bg_weights,
        const std::vector<double>& sig_targets,
        const std::vector<double>& bg_targets,
        const int depth=0) const;

};


void export_dtlearner ();



#endif  /* PYBDT_DTLEARNER_HPP */
