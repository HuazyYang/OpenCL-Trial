#pragma once

//Dynamically strength-reduced div and mod
//
//Ideas taken from Sean Baxter's MGPU library.
//These classes provide for reduced complexity division and modulus
//on integers, for the case where the same divisor or modulus will
//be used repeatedly.  

namespace tr_inplace {

namespace details {


void find_divisor(unsigned int denom,
                  unsigned int& mul_coeff, unsigned int& shift_coeff);


void find_divisor(unsigned long long denom,
                  unsigned long long& mul_coeff, unsigned int& shift_coeff);


inline unsigned int umulhi(unsigned int x, unsigned int y) {
    unsigned long long z = (unsigned long long)x * (unsigned long long)y;
    return (unsigned int)(z >> 32);
}


unsigned long long host_umulhi(unsigned long long x, unsigned long long y);

inline unsigned long long umulhi(unsigned long long x, unsigned long long y) {
    return host_umulhi(x, y);
}

}

template<typename U>
struct reduced_divisor_impl {
    U mul_coeff;
    unsigned int shift_coeff;
    U y;

    reduced_divisor_impl(U _y) : y(_y) {
        details::find_divisor(y, mul_coeff, shift_coeff);
    }

    U div(U x) const {
        return (mul_coeff) ? details::umulhi(x, mul_coeff) >> shift_coeff : x;
    }

    U mod(U x) const {
        return (mul_coeff) ? x - (div(x) * y) : 0;
    }

    void divmod(U x, U& q, U& mod) {
        if (y == 1) {
            q = x; mod = 0;
        } else {
            q = div(x);
            mod = x - (q * y);
        }
    }   

    U get() const {
        return y;
    }
};

typedef reduced_divisor_impl<unsigned int> reduced_divisor;
typedef reduced_divisor_impl<unsigned long long> reduced_divisor_64;


}
