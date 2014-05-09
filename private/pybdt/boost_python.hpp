// boost_python.hpp
//
// Without the 2 #undef's below, I get warnings on every instance of
// #include <boost/python.hpp> on my system

#ifndef PYBDT_BOOST_PYTHON_HPP
#define PYBDT_BOOST_PYTHON_HPP

#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <boost/python.hpp>

#endif  /* PYBDT_BOOST_PYTHON_HPP */
