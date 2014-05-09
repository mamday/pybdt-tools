// random_sampler.cpp


#include "random_sampler.hpp"



RandomSampler::RandomSampler (int seed)
{
    gsl_rng_env_setup ();
    const gsl_rng_type* T (gsl_rng_default);
    m_rng = gsl_rng_alloc (T);
    gsl_rng_set (m_rng, seed);
}
