#pragma once

#include <vector>

namespace m_height_lp {

// Start with all -1 signs. Advance in binary-mask order, first sign fastest,
// without a fixed-width integer. Exhaustion resets the signs and returns false.
inline bool advance_sign_pattern(std::vector<int>& signs) {
    for (int& sign : signs) {
        if (sign == -1) {
            sign = 1;
            return true;
        }
        sign = -1;
    }
    return false;
}

} // namespace m_height_lp
