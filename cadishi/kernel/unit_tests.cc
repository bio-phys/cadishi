#include <cstring>
#include <cstdio>
#include <cmath>
#include <stdexcept>

#include <pydh.h>
#include <common.hpp>

template <typename TUPLE3_T, typename FLOAT_T>
int test_triclinic_box()
{
    TUPLE3_T box[3];
    TUPLE3_T box_inv[3];
    memset(box, 0, 3*sizeof(box[0]));
    memset(box_inv, 0, 3*sizeof(box[0]));

    // trivial box
    // box[0].x = 0.67;
    // box[1].y = 0.67;
    // box[2].z = 0.67;

    // arbitrary triclinic box
    box[0].x = 0.71;

    box[1].x = 0.63;
    box[1].y = 0.42;

    box[2].x = 0.51;
    box[2].y = 0.67;
    box[2].z = 0.28;

    calclulate_inverse_triclinic_box<TUPLE3_T, FLOAT_T>(box, box_inv);

    TUPLE3_T tup = {0.0};
    tup.x = 1.0;
    tup.y = 1.0;
    tup.z = 1.0;
    TUPLE3_T tup_orig = tup;

    // printf("cartesian: %f %f %f\n", tup.x, tup.y, tup.z);

    transform_to_triclinic_coordinates<TUPLE3_T, FLOAT_T>(tup, box_inv);

    // printf("triclinic: %f %f %f\n", tup.x, tup.y, tup.z);

    transform_to_cartesian_coordinates<TUPLE3_T, FLOAT_T>(tup, box);

    // printf("cartesian: %f %f %f\n", tup.x, tup.y, tup.z);

    FLOAT_T delta = std::sqrt(  std::pow(tup.x - tup_orig.x, 2)
                              + std::pow(tup.y - tup_orig.y, 2)
                              + std::pow(tup.z - tup_orig.z, 2));

    FLOAT_T eps = 5.e-7;
    // printf("%f\n", delta);

    if (delta > eps) {
        throw std::runtime_error("transform bogus");
    }
}

int main() {
    try {
        test_triclinic_box<tuple3s_t, float>();
    } catch (...) {
        return 1;
    }
    return 0;
}
