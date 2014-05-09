// bdtlearner.cpp

#include <Python.h>

#include "vinelearner.hpp"
#include "np.hpp"

#include "notifier.hpp"

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/pointer_cast.hpp>

#include "vinemodel.hpp"


using namespace std;
using namespace boost;
namespace py = boost::python;


VineLearner::VineLearner (const string& vine_feature,
                          const double vine_feature_min,
                          const double vine_feature_max,
                          const double vine_feature_width,
                          const double vine_feature_step,
                          boost::shared_ptr<Learner> learner)
: Learner (learner->feature_names (),
           learner->sig_weight_name (), learner->bg_weight_name ()),
    m_vine_feature (vine_feature),
    m_vine_feature_min (vine_feature_min),
    m_vine_feature_max (vine_feature_max),
    m_vine_feature_width (vine_feature_width),
    m_vine_feature_step (vine_feature_step),
    m_learner (learner)
{
    for (size_t i (0); i < m_feature_names.size (); ++i) {
        if (m_feature_names[i] == m_vine_feature) {
            m_vine_feature_i = i;
            break;
        }
    }
}

VineLearner::~VineLearner ()
{
}


bool
VineLearner::quiet () const {
    return m_quiet;
}

string
VineLearner::vine_feature () const
{
    return m_vine_feature;
}

double
VineLearner::vine_feature_min () const
{
    return m_vine_feature_min;
}

double
VineLearner::vine_feature_max () const
{
    return m_vine_feature_max;
}

double
VineLearner::vine_feature_width () const
{
    return m_vine_feature_width;
}

double
VineLearner::vine_feature_step () const
{
    return m_vine_feature_step;
}

boost::shared_ptr<Learner>
VineLearner::learner ()
{
    return m_learner;
}


void
VineLearner::quiet (bool val) {
    m_quiet = val;
}

void
VineLearner::vine_feature (const string& vine_feature)
{
    m_vine_feature = vine_feature;
}

void
VineLearner::vine_feature_min (const double vine_feature_min)
{
    m_vine_feature_min = vine_feature_min;
}

void
VineLearner::vine_feature_max (const double vine_feature_max)
{
    m_vine_feature_max = vine_feature_max;
}

void
VineLearner::vine_feature_width (const double vine_feature_width)
{
    m_vine_feature_width = vine_feature_width;
}

void
VineLearner::vine_feature_step (const double vine_feature_step)
{
    m_vine_feature_step = vine_feature_step;
}


boost::shared_ptr<Model>
VineLearner::train_given_everything (
    const vector<Event>& sig, const vector<Event>& bg,
    const vector<double>& init_sig_weights,
    const vector<double>& init_bg_weights) const
{
    // some types
    typedef vector<boost::shared_ptr<Model> > vecm;
    typedef vector<Event> vecev;
    typedef vector<double> vecd;
    typedef vecev::const_iterator vecev_citer;
    typedef vecd::const_iterator vecd_citer;

    // reusable iterators
    vecev_citer i_ev;
    vecd_citer i_weight;

    // normalize weights
    vecd sig_weights (np::div (init_sig_weights, np::sum (init_sig_weights)));
    vecd bg_weights (np::div (init_bg_weights, np::sum (init_bg_weights)));

    vecd bin_mins;
    vecd bin_maxs;
    vecm models;
    for (double feature_min (m_vine_feature_min);
         feature_min + m_vine_feature_width <= m_vine_feature_max;
         feature_min += m_vine_feature_step) {
        double feature_max (feature_min + m_vine_feature_width);
        if (not m_quiet) {
            cout << "Working on " << feature_min
                << " <= " << m_vine_feature
                << " < " << feature_max
                << "..." << endl;
        }
        vecev bin_sig;
        vecd bin_sig_weights;
        vecev bin_bg;
        vecd bin_bg_weights;
        bin_sig.reserve (sig.size ());
        bin_sig_weights.reserve (sig.size ());
        bin_bg.reserve (bg.size ());
        bin_bg_weights.reserve (bg.size ());
        for (i_ev = sig.begin (), i_weight = sig_weights.begin ();
             i_ev != sig.end (); ++i_ev, ++i_weight) {
            double value (i_ev->value (m_vine_feature_i));
            if ((feature_min <= value) and (value < feature_max)) {
                bin_sig.push_back (*i_ev);
                bin_sig_weights.push_back (*i_weight);
            }
        }
        for (i_ev = bg.begin (), i_weight = bg_weights.begin ();
             i_ev != bg.end (); ++i_ev, ++i_weight) {
            double value (i_ev->value (m_vine_feature_i));
            if ((feature_min <= value) and (value < feature_max)) {
                bin_bg.push_back (*i_ev);
                bin_bg_weights.push_back (*i_weight);
            }
        }
        boost::shared_ptr<Model> bin_model (m_learner->train_given_everything (
                bin_sig, bin_bg, bin_sig_weights, bin_bg_weights));
        bin_mins.push_back (feature_min);
        bin_maxs.push_back (feature_max);
        models.push_back (bin_model);
    }
    return boost::make_shared<VineModel> (
        m_feature_names, m_vine_feature, bin_mins, bin_maxs, models);
}


void export_vinelearner ()
{
    using namespace boost::python;

    class_<VineLearner, bases<Learner> > (
        "VineLearner",
        init<string,double,double,double,double,boost::shared_ptr<Learner> > ())
        .add_property (
            "vine_feature",
            (string (VineLearner::*)()const) &VineLearner::vine_feature,
            (void (VineLearner::*)(const string&)) &VineLearner::vine_feature)
        .add_property (
            "vine_feature_min",
            (double (VineLearner::*)()const) &VineLearner::vine_feature_min,
            (void (VineLearner::*)(double)) &VineLearner::vine_feature_min)
        .add_property (
            "vine_feature_max",
            (double (VineLearner::*)()const) &VineLearner::vine_feature_max,
            (void (VineLearner::*)(double)) &VineLearner::vine_feature_max)
        .add_property (
            "vine_feature_width",
            (double (VineLearner::*)()const) &VineLearner::vine_feature_width,
            (void (VineLearner::*)(double)) &VineLearner::vine_feature_width)
        .add_property (
            "vine_feature_step",
            (double (VineLearner::*)()const) &VineLearner::vine_feature_step,
            (void (VineLearner::*)(double)) &VineLearner::vine_feature_step)
        .add_property (
            "learner", &VineLearner::learner)
        ;
}
