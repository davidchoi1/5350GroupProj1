#include <omp.h> // OpenMP header for parallel programming
#include <iostream>


/**
 * For the sake of simplicity, we will implement the algorithms as well as the correctness tests in one 
 * long spaghetti file (this current file - open_mp.cpp). Below is a rough draft of function prototypes
 * and their implementations (to be implemented)
*/


// Function prototypes
void serialMM(...);
void parallelMM(...);
void oneDMM(...);
void twoDMM(...);
void testMM();

/**
 * Right now, this main method is inhabited by a very basic, multi-thread hello world print check, to 
 * test if openMP works in your environment. 
*/
int main() {
    // OpenMP pragma directive to create a parallel region
    #pragma omp parallel
    {
        // Get the unique identifier of the current thread
        int ID = omp_get_thread_num();

        // Print a message indicating which thread is executing
        std::cout << "Hello from thread " << ID << std::endl;
    }
    // End of the parallel region

    return 0;
}


void serialMM(...) { /* Implementation */ }
void parallelMM(...) { /* Implementation */ }
void oneDMM(...) { /* Implementation */ }
void twoDMM(...) { /* Implementation */ }


// This is a correctness test, not a performance test
void testMM() {
    // Call MM functions with test cases
    // Compare results to expected outcomes
    // Report any discrepancies
}