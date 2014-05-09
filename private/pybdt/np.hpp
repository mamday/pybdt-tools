// np.hpp

#ifndef PYBDT_NP_HPP
#define PYBDT_NP_HPP

#include <cstdlib>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

#include "boost_python.hpp"

#define PY_ARRAY_UNIQUE_SYMBOL pybdt_ARRAY_API
#define NO_IMPORT_ARRAY

#include <numpy/ndarrayobject.h>

namespace np {   // vector utility functions

using namespace std;
using namespace boost::python;

template <typename T>
vector<T>
array_to_vector (PyObject* a)
{
    int N = PyArray_Size (a);
    vector<T> out (N);
    for (int i = 0; i < N; ++i) {
        out[i] = T (*((T *) PyArray_GETPTR1 (a, i)));
    }
    return out;
}

template <typename T>
std::vector<T>
list_to_vector (const boost::python::list& l)
{
    int N = len (l);
    vector<T> out (N);
    for (int i = 0; i < N; ++i) {
        out[i] = extract<T> (l[i]);
    }
    return out;
}

template <typename T>
boost::python::list
vector_to_list (const vector<T>& v)
{
    typedef typename std::vector<T>::const_iterator citer;
    boost::python::list out;
    for (citer i = v.begin (); i != v.end (); ++i) {
        out.append (*i);
    }
    return out;
}

template <typename T>
PyObject*
vector_to_array (const vector<T>& v)
{
    typedef typename std::vector<T>::size_type size_type;
    size_type N (v.size ());
    npy_intp dims[1];
    dims[0] = N;
    PyObject* C_a (PyArray_SimpleNew (1, dims, NPY_FLOAT64));
    for (size_type i(0); i < N; ++i) {
        double* dest ((double*) PyArray_GETPTR1 (C_a, i));
        *dest = v[i];
    }
    return C_a;
}

template <typename T, typename T2>
vector<T>
range (T2 lower, T2 upper=0)
{
    if (upper == 0) {
        upper = lower;
        lower = 0;
    }
    vector<T> out (upper - lower);
    for (T2 i = lower; i < upper; ++i) {
        out[i] = i;
    }
    return out;
}

template <typename T>
vector<T>
ones (int n)
{
    vector<T> out (n);
    for (int i = 0; i < n; ++i) {
        out[i] = 1;
    }
    return out;
}

template <typename T>
vector<T>
zeros (int n)
{
    vector<T> out (n);
    for (int i = 0; i < n; ++i) {
        out[i] = 0;
    }
    return out;
}

template <typename T>
T
sum (const vector<T>& v)
{
    typedef typename vector<T>::const_iterator citer;
    T out (0);
    for (citer iv = v.begin (); iv != v.end (); ++iv) {
        out += *iv;
    }
    return out;
    // return std::accumulate (v.begin (), v.end (), 0, std::plus<T>());
}

template <typename T>
T
max (const vector<T>& v)
{
    int n (v.size ());
    assert (n > 0);
    T out (v[0]);
    for (int i = 1; i < n; ++i) {
        if (v[i] > out) {
            out = v[i];
        }
    }
    return out;
}

template <typename T>
T
max (const vector<T>& v, const vector<int>& idx)
{
    int n (idx.size ());
    assert (n > 0);
    assert (v.size () > 0);
    T out (v[idx[0]]);
    for (int i = 1; i < n; ++i) {
        if (v[idx[i]] > out) {
            out = v[idx[i]];
        }
    }
    return out;
}

template <typename T>
T
min (const vector<T>& v)
{
    int n (v.size ());
    assert (n > 0);
    T out (v[0]);
    for (int i = 1; i < n; ++i) {
        if (v[i] < out) {
            out = v[i];
        }
    }
    return out;
}

template <typename T>
T
min (const vector<T>& v, const vector<int>& idx)
{
    int n (idx.size ());
    assert (n > 0);
    assert (v.size () > 0);
    T out (v[idx[0]]);
    for (int i = 1; i < n; ++i) {
        if (v[idx[i]] < out) {
            out = v[idx[i]];
        }
    }
    return out;
}

template <typename T>
int
index (const vector<T>& v, const T& t)
{
    int n = v.size ();
    for (int i = 0; i < n; ++i) {
        if (v[i] == t) {
            return i;
        }
    }
    return -1;
}

template <typename T, typename T2>
vector<bool>
lt (const vector<T>& v, const T2& t)
{
    const int n = v.size ();
    vector<bool> out (n);
    for (int i = 0; i < n; ++i) {
        out[i] = (v[i] < t);
    }
    return out;
}

template <typename T, typename T2>
vector<bool>
gt (const vector<T>& v, const T2& t)
{
    const int n = v.size ();
    vector<bool> out (n);
    for (int i = 0; i < n; ++i) {
        out[i] = (v[i] > t);
    }
    return out;
}

template <typename T, typename T2>
vector<bool>
lte (const vector<T>& v, const T2& t)
{
    const int n = v.size ();
    vector<bool> out (n);
    for (int i = 0; i < n; ++i) {
        out[i] = (v[i] <= t);
    }
    return out;
}

template <typename T, typename T2>
vector<bool>
gte (const vector<T>& v, const T2& t)
{
    const int n = v.size ();
    vector<bool> out (n);
    for (int i = 0; i < n; ++i) {
        out[i] = (v[i] >= t);
    }
    return out;
}

template <typename T1, typename T2>
bool
dims_match (const vector<T1>& a, const vector<T2>& b)
{
    return a.size () == b.size ();
}

template <typename func, typename T>
vector<typename func::result_type>
operate (const vector<T>& a, const vector<T>& b)
{
    assert (dims_match (a, b));
    vector<typename func::result_type> out (a.size ());
    typedef typename vector<T>::iterator       iter;
    typedef typename vector<T>::const_iterator citer;
    citer   ia = a.begin (), ib = b.begin ();
    iter    iout = out.begin ();
    func    f = func ();
    for (; iout != out.end (); ++ia, ++ib, ++iout) {
        *iout = f (*ia, *ib);
    }
    return out;
}

template <typename func, typename T, typename T2>
vector<typename func::result_type>
operate (const vector<T>& a, const T2& b)
{
    vector<typename func::result_type> out (a.size ());
    typedef typename vector<T>::iterator       iter;
    typedef typename vector<T>::const_iterator citer;
    typedef typename vector<typename func::result_type>::iterator oiter;
    citer   ia = a.begin ();
    oiter   iout = out.begin ();
    func    f = func ();
    for (; iout != out.end (); ++ia, ++iout) {
        *iout = f (*ia, b);
    }
    return out;
}

template <typename T, typename T2>
vector<T>
add (const vector<T>& a, const T2& b)
{
    return operate<std::plus<T> > (a, b);
}

template <typename T, typename T2>
vector<T>
sub (const vector<T>& a, const T2& b)
{
    return operate<std::minus<T> > (a, b);
}

template <typename T, typename T2>
vector<T>
mul (const vector<T>& a, const T2& b)
{
    return operate<std::multiplies<T> > (a, b);
}

template <typename T, typename T2>
vector<T>
div (const vector<T>& a, const T2& b)
{
    return operate<std::divides<T> > (a, b);
}

template <typename T, typename T2>
vector<bool>
eq (const vector<T>& a, const T2& b)
{
    return operate<std::equal_to<T> > (a, b);
}

template <typename T>
T
prod (const vector<T>& v)
{
    return std::accumulate (v.begin (), v.end (), T (1),
                            std::multiplies<T>());
}

template <typename T>
vector<T>
exp (const vector<T>& v)
{
    typedef typename vector<T>::iterator iter;
    typedef typename vector<T>::const_iterator citer;
    vector<T> out (v.size ());
    iter iout = out.begin ();
    citer iv = v.begin ();
    for (; iv != v.end (); ++iv, ++iout) {
        *iout = std::exp (*iv);
    }
    return out;
}

template <typename T>
vector<bool>
isfinite (const vector<T>& v)
{
    typedef typename vector<T>::const_iterator v_citer;
    typedef typename vector<bool>::iterator out_iter;
    vector<bool> out (v.size ());
    out_iter iout = out.begin ();
    v_citer iv = v.begin ();
    for (; iv != v.end (); ++iv, ++iout) {
        *iout = std::isfinite (*iv);
    }
    return out;
}

template <typename Tout, typename T>
vector<Tout>
conv (const vector<T>& v)
{
    vector<Tout> out (v.size ());
    typedef typename vector<Tout>::iterator iter_out;
    typedef typename vector<T>::const_iterator citer_in;
    iter_out iout = out.begin ();
    citer_in iv = v.begin ();
    for (; iv != v.end (); ++iv, ++iout) {
        *iout = Tout (*iv);
    }
    return out;
}

template <typename T1, typename T2>
vector<T1>
subscript (const vector<T1> v, const vector<T2> indices)
{
    vector<T1> out;
    out.reserve (indices.size ());
    for (typename vector<T2>::const_iterator i_index (indices.begin ());
         i_index != indices.end (); ++i_index) {
        out.push_back (v[*i_index]);
    }
    return out;
}

template <typename T>
vector<T>
subscript (const vector<T> v, const vector<bool> keep)
{
    // v and keep are same size; keep v element if keep element is true
    typedef typename vector<T>::const_iterator v_citer;
    typedef typename vector<bool>::const_iterator keep_citer;
    assert (dims_match (v, keep));
    vector<T> out;
    out.reserve (v.size ());
    v_citer i_v (v.begin ());
    keep_citer i_keep (keep.begin ());
    for (; i_v != v.end (); ++i_v, ++i_keep) {
        if (*i_keep) {
            out.push_back (*i_v);
        }
    }
    return out;
}

template <typename T>
std::string
stringify (const vector<T> v)
{
    using namespace std;
    typedef typename vector<T>::const_iterator v_citer;
    ostringstream out;
    out << "[ ";
    int count (0);
    for (v_citer i (v.begin ()); i != v.end (); ++i, ++count) {
        out << setiosflags(ios::fixed) << setprecision(4) << scientific
            << setw (18) << *i;
        if (count % 4 == 0) {
            out << endl << "  ";
        }
    }
    out << " ]";
    return out.str ();
}



}


#endif  // PYBDT_NP_HPP
