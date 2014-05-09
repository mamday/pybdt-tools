// dataset.hpp

#ifndef PYBDT_DATASET_HPP
#define PYBDT_DATASET_HPP


#include <cmath>
#include <iostream>
#include <vector>
#include <string>

#include "boost_python.hpp"

class DataSet;

class Event {
    typedef std::vector<double>::size_type size_type;
public:
    // structors
    Event (const DataSet* dataset, const int row);

    Event& operator= (const Event& e);

    // inspectors

    double operator[] (const size_type index) const;

    bool all_finite () const;
    int get_column_index (const std::string& name) const;
    size_type size () const;
    double value (const size_type index) const;
    std::vector<double> values () const;

private:
    const DataSet* m_dataset;
    int m_row;
};

class DataSet {
    friend class Event;
public:

    // structors

    explicit DataSet (const boost::python::dict& data,
                      const std::string subset="all");

    DataSet (const DataSet& other,
             const std::vector<std::string>& names);


    // inspectors

    const Event& event (int index) const;
    const std::vector<Event>& events () const;

    std::vector<double> get_column (const std::string& name) const;
    int get_column_index (const std::string& name) const;
    std::vector<int> get_column_indices (
        const std::vector<std::string>& feature_names) const;
    PyObject* getitem (const std::string& name) const;

    double livetime () const;

    std::vector<std::string> names () const;
    boost::python::list names_py () const;

    int n_features () const;
    int n_events () const;



    // conversions

    boost::python::dict to_dict () const;


    // mutators

    void livetime (const double t);


protected:

    std::vector<std::string> m_names;
    std::vector<Event> m_events;

    int m_n_features;
    int m_n_events;
    double m_livetime;

    std::vector<std::vector<double> > m_cols;
};


void export_dataset ();



inline double
Event::operator[] (const size_type index) const
{
    return value (index);
}

inline bool
Event::all_finite () const
{
    size_type n (size ());
    for (size_type j_col (0); j_col < n; ++j_col) {
        if (not std::isfinite (value (j_col))) {
            return false;
        }
    }
    return true;
}

inline int
Event::get_column_index (const std::string& name) const
{
    return m_dataset->get_column_index (name);
}

inline std::vector<double>::size_type
Event::size () const
{
    return m_dataset->n_features ();
}

inline double
Event::value (const size_type index) const
{
    return m_dataset->m_cols[index][m_row];
}

inline std::vector<double>
Event::values () const
{
    std::vector<double> out (size ());
    for (unsigned j_col (0); j_col < out.size (); ++j_col) {
        out[j_col] = value (j_col);
    }
    return out;
}


inline const Event&
DataSet::event (int index) const
{
    return m_events.at (index);
}

inline const std::vector<Event>&
DataSet::events () const
{
    return m_events;
}

inline int
DataSet::n_features () const
{
    return m_n_features;
}

inline int
DataSet::n_events () const
{
    return m_n_events;
}


#endif  /* PYBDT_DATASET_HPP */
