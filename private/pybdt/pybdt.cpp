// pybdt.cpp

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

#include <boost/assign/list_of.hpp>
#include <boost/assign/std/vector.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/tuple/tuple.hpp>

#include "boost_python.hpp"

#define PY_ARRAY_UNIQUE_SYMBOL pybdt_ARRAY_API
#include <numpy/ndarrayobject.h>

#include "np.hpp"
#include "dataset.hpp"
#include "model.hpp"
#include "learner.hpp"
#include "dtmodel.hpp"
#include "dtlearner.hpp"
#include "bdtmodel.hpp"
#include "bdtlearner.hpp"
#include "vinemodel.hpp"
#include "vinelearner.hpp"


using namespace std;
using namespace boost;
using namespace boost::python;
namespace py = boost::python;

#if PY_MAJOR_VERSION >= 3
static PyObject *hack_import_array() {import_array(); return NULL;}
#else
static void hack_import_array() {import_array();}
#endif

BOOST_PYTHON_MODULE (_pybdt)
{
    hack_import_array ();

    export_dataset ();
    export_model ();
    export_dtmodel ();
    export_bdtmodel ();
    export_vinemodel ();
    export_learner ();
    export_dtlearner ();
    export_bdtlearner ();
    export_vinelearner ();
    export_pruners ();
}
