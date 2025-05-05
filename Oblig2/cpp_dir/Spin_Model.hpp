#include <iostream>
#include <vector>
#include <cstdlib>
#include <math.h>
#include <complex>

// using namespace std;

#ifndef __Spin_Model_hpp__
#define __Spin_Model_hpp__

class Spin_Model {
    private:
        int q; // Number of spin states
        int d; // Dimension of system. Should be 1 or 2 (1D chain or 2D lattice)
        int L; // Number of sites along each axis
        int N; // System size (total number of sites)

        double T; // Temperature in units of J
        double P_connect; // Connection probability

        std::vector<int> S; // Spin state of each site
        std::vector< std::complex<double> > m_spin; // Possible values of m_j from spin state

        std::vector<int> M_count; // Magnetization counter
    
    public:
        Spin_Model(int d, int L, double T);
        // int neighbour_index(int i, int dir);
        // void flip_and_build(int i);
        // double corr_func_anal(double r);
        // std::map<std::string, arma::vec> run_MC(int n_clusters, int n_steps_eq, int n_steps);
};

#endif