#include<iostream>
#include<vector>
#include<cstdlib>
#include<math.h>
#include<complex>

using namespace std ;

#define PI 3.141592653589793238462643

const int q = 3; // q spin states
const int L = 16; // Linear system size
const double T = 0.25; // Temperature in units of J

const int N = L ; // Total number of spins
const double pconnect = 0.5; // Connection probability

const int NCLUSTERS = 1; // Number of cluster builds in one MC step.
const int NESTEPS = 10000; // Number of equilibrium MC steps.
const int NMSTEPS = 10000; // Number of measurement MC step.
const int NBINS = 10; // Number of measurement bins

vector<int> S(N); // The spin array
vector<int> M(q); // Number of spins in different states.
vector< complex<double> > W(q); // Order parameter weights

// Lattice handling:
enum dirs{RIGHT, LEFT};
int indx(int x) {return x;} // Make an index on every site
int xpos(int i) {return i%L;}

int Nbr(int i, int dir) {
    int x = xpos(i);
    switch(dir) {
        case RIGHT: return indx((x+1)%L);
        case LEFT: return indx((x-1+L)%L);
    }
    cout << "ERROR: dir must be LEFT or RIGHT!." << endl;
    throw exception();
}

void FlipandBuildFrom(int s) {
    int oldstate(S[s]), newstate((S[s]+1)%q);

    S[s] = newstate; // Flipspin
    M[oldstate]--; M[newstate]++; // Update spin counts
    
    for (int dir=0; dir<2; dir++) { // Go through neighbors
        int j=Nbr(s, dir);
        if (S[j] == oldstate)
            if (rand() / (RAND_MAX + 1.) < pconnect) {
                FlipandBuildFrom(j);
            }
    }
}

int main () {
    // Initialize order parameter weights
    for (int s=0; s<q; s++)
        W[s] = complex<double>(cos(2*PI*s/q), sin(2*PI*s/q));
    for (int i=0; i<N; i++)
        S[i]=0; // Initialize to the spin=0 state
    for (int s=1; s<q; s++)
        M[s]=0; // Initialize counters.

    M[0] = N;
    srand((unsigned) time(0)); // Initialize random number gen.

    // Equilibriate
    for (int t=0; t<NESTEPS; t++)
        for (int c=0; c<NCLUSTERS; c++) {
            FlipandBuildFrom(rand() % N);
    }
    
    cout << "# m abs(m) abs(m)^2 abs(m)^4" << endl;

    // Measure
    for (int n=0; n<NBINS; n++) {
        complex<double> m(0., 0.);
        double m1=0, m2=0, m4=0; // Measurement results
        
        for (int t=0; t<NMSTEPS; t++) {
            for (int c=0; c<NCLUSTERS; c++)
                FlipandBuildFrom(rand() % N);
            
            complex<double> tm(0., 0.);
            
            for (int s=0; s<q; s++) {
                tm += W[s] * double (M[s]);
            }

            tm /= N;
            double tm1 = abs(tm);
            double tm2 = tm1 * tm1;
            m += tm;
            m1 += tm1;
            m2 += tm2;
            m4 += tm2 * tm2;
        }

        m /= NMSTEPS; m1 /= NMSTEPS; m2 /= NMSTEPS; m4 /= NMSTEPS;
        
        // Output in a numpy-friendly format
        cout << m.real() << "+" << m.imag() << "j " << m1 << " " << m2 << " " << m4 << endl;
    }
}