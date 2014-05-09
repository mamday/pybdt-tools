// random_sampler.hpp
// randomly sample arrays


#ifndef PYBDT_RANDOM_SAMPLER_HPP
#define PYBDT_RANDOM_SAMPLER_HPP

#include <set>
#include <vector>

#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

#include "np.hpp"

class RandomSampler {
public:
    RandomSampler (int seed = 0);

    template <typename T>
    std::vector<T> sample (
        const unsigned n, const std::vector<T>& vec, const bool replace=false)
    {
        using namespace std;
        const vector<int> indices (sample_range<T> (
                n, 0, vec.size (), replace));
        return np::subscript (vec, indices);
    }

    template <typename T, typename T2>
    std::vector<T> sample_range (
        const unsigned n, const T2& i1, const T2& i2=0,
        const bool replace=false)
    {
        using namespace std;
        assert (i2 > i1);
        const T2 len (i2 - i1);
        vector<T> out;
        out.reserve (n);
        set<unsigned> already_picked;
        for (unsigned i_pick (0); i_pick < n; ++i_pick) {
            unsigned pick = gsl_rng_uniform_int (m_rng, len) + i1;
            if (not replace) {
                // if not replacing, keep picking till a new one is found
                while (already_picked.find (pick) != already_picked.end ()) {
                    pick = gsl_rng_uniform_int (m_rng, len) + i1;
                }
                already_picked.insert (pick);
            }
            out.push_back (pick);
        }
        return out;
    }

private:

    gsl_rng* m_rng;

};

#endif  /* PYBDT_RANDOM_SAMPLER_HPP */
