// dataset.cpp


#include "dataset.hpp"

#include "np.hpp"

#include <cmath>

#include <boost/make_shared.hpp>

#define PY_ARRAY_UNIQUE_SYMBOL pybdt_ARRAY_API
#define NO_IMPORT_ARRY
#include <numpy/ndarrayobject.h>


using namespace std;
using namespace boost;
using namespace boost::python;
namespace py = boost::python;


// Event ------------------------------------------------------------

Event::Event (const DataSet* dataset, const int row)
    : m_dataset (dataset), m_row (row)
{ }

Event&
Event::operator= (const Event& e)
{
    m_dataset = e.m_dataset;
    m_row = e.m_row;
    return *this;
}


// DataSet ----------------------------------------------------------

DataSet::DataSet (const py::dict& data, string subset)
: m_n_features (0), m_n_events (0), m_livetime (-1)
{
    // data should be a dict of (str, numpy.ndarray) items
    int n_input_cols (len (data));
    py::list keys = data.keys ();
    keys.sort ();
    vector<vector<double> > cols;
    // get data into vectors
    for (int j = 0; j < n_input_cols; ++j) {
        string name = extract<string> (keys[j]);
        if (name == "livetime") {
            m_livetime = extract<double> (data[name]);
            continue;
        }
        py::object pycol = data[name];
        PyObject* C_numpy_col = pycol.ptr ();
        if (not PyArray_Check (C_numpy_col)) {
            continue;
        }
        m_names.push_back (name);
        // n.b.: C_numpy_col had better consist of floats!
        // (pure-Python wrapper layer ensures this)
        cols.push_back (np::array_to_vector<double> (C_numpy_col));
    }
    m_n_features = cols.size ();
    m_n_events = cols[0].size ();
    // check for matching column lengths
    for (int i_col (1); i_col < m_n_features; ++i_col) {
        if (int (cols[i_col].size ()) != m_n_events) {
            throw runtime_error (
                "column " + m_names[i_col] + "had non-matching length");
        }
    }
    // set up permanent columns, events
    m_cols.resize (m_n_features);
    int n_reserve;
    if ((subset == "even") or (subset == "odd")) {
        n_reserve = m_n_events / 2 + 2;
    }
    else {
        n_reserve = m_n_events;
    }
    for (int i_col (0); i_col < m_n_features; ++i_col) {
        m_cols[i_col].reserve (n_reserve);
    }
    m_events.reserve (n_reserve);
    int i_row_kept (0);
    for (int i_row (0); i_row < m_n_events; ++i_row) {
        if ((subset == "even") and (i_row % 2 == 1)) {
            continue;
        }
        if ((subset == "odd") and (i_row % 2 == 0)) {
            continue;
        }
        for (int j_col (0); j_col < m_n_features; ++j_col) {
            m_cols[j_col].push_back (cols[j_col][i_row]);
        }
        m_events.push_back (Event (this, i_row_kept));
        i_row_kept += 1;
    }
    if ((subset == "even") or (subset == "odd")) {
        m_n_events = m_events.size ();
    }
}


DataSet::DataSet (const DataSet& other,
                  const vector<string>& names)
: m_names (names), m_n_features (names.size ()), m_livetime (other.m_livetime)
{
    // info about other
    vector<int> indices = other.get_column_indices (m_names);
    int n_other (other.m_n_events);
    m_events.reserve (n_other);
    // set up permanent columns, events
    m_cols.resize (m_n_features);
    for (int i_col (0); i_col < m_n_features; ++i_col) {
        m_cols[i_col].reserve (n_other);
    }
    m_events.reserve (n_other);
    // copy data
    int i_row (0);
    for (; i_row < n_other; ++i_row) {
        const Event& ev_other = other.event (i_row);
        for (int j_col (0); j_col < m_n_features; ++j_col) {
            m_cols[j_col].push_back (ev_other.value (indices[j_col]));
        }
        m_events.push_back (Event (this, i_row));
    }
    m_n_events = m_events.size ();
}


vector<double>
DataSet::get_column (const string& name) const
{
    const int index (get_column_index (name));
    return m_cols[index];
}

int
DataSet::get_column_index (const string& name) const
{
    int index = np::index (m_names, name);
    if (index >= 0) {
        return index;
    }
    else {
        throw runtime_error (
            "DataSet does not contain \"" + name + "\"");
    }
}

vector<int>
DataSet::get_column_indices (
    const vector<string>& feature_names) const
{
    using namespace std;
    vector<int> out;
    for (vector<string>::const_iterator i = feature_names.begin ();
         i != feature_names.end (); ++i) {
        int index = np::index (m_names, *i);
        if (index >= 0) {
            out.push_back (index);
        }
        else {
            throw runtime_error (
                "DataSet does not contain \"" + *i + "\"");
        }
    }
    return out;
}

PyObject*
DataSet::getitem (const string& name) const
{
    return np::vector_to_array (get_column (name));
}

inline double
DataSet::livetime () const
{
    return m_livetime;
}

vector<string>
DataSet::names () const
{
    return m_names;
}

py::list
DataSet::names_py () const
{
    return np::vector_to_list (m_names);
}

py::dict
DataSet::to_dict () const
{
    dict out;
    for (vector<string>::const_iterator i_name = m_names.begin ();
         i_name != m_names.end (); ++i_name) {
        out[*i_name] = object (handle<> (borrowed (getitem (*i_name))));
    }
    return out;
}

inline void
DataSet::livetime (const double t)
{
    m_livetime = t;
}


void
export_dataset ()
{

    class_<DataSet> (
        "DataSet",
         init<py::dict> ())
        .def (init<py::dict,string> ())
        .def ("get_column", &DataSet::get_column)
        .def ("to_dict", &DataSet::to_dict)
        .def ("__getitem__", &DataSet::getitem)
        .def ("__len__", &DataSet::n_events)
        .add_property ("n_features", &DataSet::n_features)
        .add_property ("n_events", &DataSet::n_events)
        .add_property ("names", &DataSet::names_py)
        .add_property (
            "livetime",
            (double (DataSet::*)()const) &DataSet::livetime,
            (void (DataSet::*)(double)) &DataSet::livetime)
        // .def_pickle (DataSet_pickle_suite ())
        ;
}
