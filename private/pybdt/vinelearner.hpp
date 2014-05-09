// vinelearner.hpp


#ifndef PYBDT_VINELEARNER_HPP
#define PYBDT_VINELEARNER_HPP

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>

#include "boost_python.hpp"

// #include "vinemodel.hpp"
#include "dataset.hpp"
#include "learner.hpp"

class VineLearner : public Learner {
    friend class VineLearner_pickle_suite;
public:

    // structors

    VineLearner (const std::string& vine_feature,
                 const double vine_feature_min,
                 const double vine_feature_max,
                 const double vine_feature_width,
                 const double vine_feature_step,
                 boost::shared_ptr<Learner> learner);

    virtual ~VineLearner ();

    // inspectors

    bool quiet () const;
    std::string vine_feature () const;
    double vine_feature_min () const;
    double vine_feature_max () const;
    double vine_feature_width () const;
    double vine_feature_step () const;

    // mutators
    void quiet (bool val);
    void vine_feature (const std::string& vine_feature);
    void vine_feature_min (const double vine_feature_min);
    void vine_feature_max (const double vine_feature_max);
    void vine_feature_width (const double vine_feature_width);
    void vine_feature_step (const double vine_feature_step);


    boost::shared_ptr<Learner> learner ();

    virtual boost::shared_ptr<Model> train_given_everything (
        const std::vector<Event>& sig, const std::vector<Event>& bg,
        const std::vector<double>& init_sig_weights,
        const std::vector<double>& init_bg_weights) const;


protected:

    std::string m_vine_feature;
    double m_vine_feature_min;
    double m_vine_feature_max;
    double m_vine_feature_width;
    double m_vine_feature_step;

    size_t m_vine_feature_i;
    bool m_quiet;

    boost::shared_ptr<Learner> m_learner;

};



void export_vinelearner ();


#endif  /* PYBDT_VINELEARNER_HPP */
