// linear_histogram.cpp


#include "linear_histogram.hpp"

#include <vector>

#include "np.hpp"


using namespace std;


Histogram::~Histogram ()
{
}

void
Histogram::fill (const vector<double>& values)
{
    this->fill (values, np::ones<double> (values.size ()));
}

