// model.hpp

#ifndef PYBDT_MODEL_HPP
#define PYBDT_MODEL_HPP

#include <cstdlib>

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "boost_python.hpp"

#include "dataset.hpp"


class Scoreable {
protected:
    typedef std::vector<double>::size_type size_type;

public:
    virtual ~Scoreable ();

    bool all_finite () const;

    virtual double operator[] (size_type i) const = 0;
    virtual size_type size () const = 0;

};

class Model {
    friend class DTModel_pickle_suite;
    friend class BDTModel_pickle_suite;
public:

    // structors

    Model (std::vector<std::string> feature_names);
    Model (boost::python::list feature_names);
    virtual ~Model ();

    // inspectors

    std::vector<std::string> feature_names () const;
    boost::python::list feature_names_py () const;

    // score methods
    double score (const Scoreable& s,
                  bool use_purity=false);

    std::vector<double> score (const DataSet& ds,
                               bool use_purity=false,
                               bool quiet=false);

    std::vector<double> score (const std::vector<Event>& events,
                               bool use_purity=false,
                               bool quiet=false);

    // score methods for use from Python
    double score_one (const boost::python::list& vals,
                      bool use_purity);
    PyObject* score_DataSet (const DataSet& ds,
                             bool use_purity,
                             bool quiet);


protected:

    virtual double base_score (const Scoreable& s,
                               bool use_purity) const = 0;

    std::vector<std::string> m_feature_names;
    int m_n_features;
};

void export_model ();



template <typename EventType>
class RealScoreable : public Scoreable {

public:

    RealScoreable (const EventType& event)
        : m_event (event)
    { }

    virtual ~RealScoreable ()
    { }

    virtual double operator[] (size_type i) const
    {
        return m_event[i];
    }

    virtual size_type size () const
    {
        return m_event.size ();
    }

private:

    const EventType& m_event;

};

template <typename EventType>
RealScoreable<EventType>
make_scoreable (const EventType& event)
{
    return RealScoreable<EventType> (event);
}


#endif  /* PYBDT_MODEL_HPP */
