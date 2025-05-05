#include "Spin_Model.hpp"

#define PI 3.141592653589793238462643

Spin_Model::Spin_Model(int d, int L, double T) {
    this->q = 3;
    this->d = d;
    this->L = L;
    this->T = T;
    this->P_connect = 1 - exp(-1.0 / T);

    if (d == 1) {
        // 1D chain
        this->N = L;
    } else {
        // 2D lattice
        this->N = L*L;
    }

    for (int i=0; i<N; i++) {
        this->S(i) = 0;
    }
    
    this->m_spin = std::vector(q, 0);
    for (int i=0; i<q; i++) {
        this->m_spin(i) = exp(2.0 * PI/3.0 * )
    }
}