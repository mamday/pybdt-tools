// notifier.hpp


#ifndef GRBLLH_NOTIFIER_HPP
#define GRBLLH_NOTIFIER_HPP


#include <iomanip>
#include <iostream>
#include <sstream>


template <typename T>
class Notifier {
public:

    Notifier (const std::string& msg, T max,
              std::ostream& stream=std::cerr, bool overwrite=true)
        : m_msg (msg), m_max (max), m_stream (stream)
    {
        using namespace std;
        ostringstream o;
        o << max;
        m_amount_width = o.str ().size ();
        if (overwrite) {
            m_first = '\r';
        }
        else {
            m_first = '\n';
        }
    }

    void update (const T& amount)
    {
        using namespace std;
        ostringstream o;
        o
            << m_first << m_msg
            << " | " << setw (m_amount_width) << amount << " of " << m_max
            << " (" << setw (6) << setprecision (2) << fixed
            << 100.0 * amount / m_max << " %)"
            << flush;
        m_last = o.str ();
        m_stream << m_last;
    }

    void finish ()
    {
        using namespace std;
        ostringstream o;
        o
            << m_first << m_msg
            << " | done.";
        while (o.str ().size () < m_last.size ()) {
            o << ' ';
        }
        m_stream << o.str () << endl;
    }


protected:

    char m_first;
    std::string m_msg;
    T m_max;
    std::ostream& m_stream;
    int m_amount_width;
    std::string m_last;

};


#endif  /* GRBLLH_NOTIFIER_HPP */
