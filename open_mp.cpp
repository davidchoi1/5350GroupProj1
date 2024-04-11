#include <omp.h> // OpenMP header for parallel programming
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib> // Random number generator
#include <ctime> 

#define DEFAULT_NUM_THREADS 9
#define NUMBER_OF_RUNS 5

/**
 * For the sake of simplicity, we will implement the algorithms as well as the correctness tests in one
 * long spaghetti file (this current file - open_mp.cpp). Below is a rough draft of function prototypes
 * and their implementations (to be implemented)
 */

// Function prototypes
std::vector<std::vector<int>> serialMM(const std::vector<std::vector<int>> &A, const std::vector<std::vector<int>> &B);

std::vector<std::vector<int>> parallelMM(const std::vector<std::vector<int>> &A, const std::vector<std::vector<int>> &B, int P);

std::vector<std::vector<int>> oneDMM(const std::vector<std::vector<int>> &A, const std::vector<std::vector<int>> &B, int P);

std::vector<std::vector<int>> twoDMM(const std::vector<std::vector<int>> &A, const std::vector<std::vector<int>> &B, int P);

std::vector<std::vector<int>> generateSquareMatrix(int);
std::vector<std::vector<int>> generateMatrix(int, int);

double testSMM(std::vector<std::vector<int>>, std::vector<std::vector<int>>, int);
void testPMM(std::vector<std::vector<int>>, std::vector<std::vector<int>>, int, int, double);
void test1DMM(std::vector<std::vector<int>>, std::vector<std::vector<int>>, int, int, double);
void test2DMM(std::vector<std::vector<int>>, std::vector<std::vector<int>>, int, int, double);

void testMM();

/**
 * Right now, this main method is inhabited by a very basic, multi-thread hello world print check, to
 * test if openMP works in your environment.
 */
int main()
{
    // Seed the random number generator
    std::srand(std::time(nullptr));
    double ts = 0.0;

    // Square Matrix of size n x n
    for(int p = 4; p <= 8; p += 4){
        for(int i = 2; i <= 10; i++){
            std::cout << i << " x " << i << " square matrix" << std::endl;
            std::cout << "Number of processors: " << p << std::endl;
            std::cout << std::endl;

        
            std::vector<std::vector<int>> A = generateSquareMatrix(i);
            std::vector<std::vector<int>> B = generateSquareMatrix(i);

            
            
            std::cout << "A matrix:" << std::endl;
            for (const auto &row : A)
            {
                for (int val : row)
                {
                    std::cout << val << " ";
                }
                std::cout << std::endl;
            }

            std::cout << std::endl;

            std::cout << "B matrix:" << std::endl;
            for (const auto &row : B)
            {
                for (int val : row)
                {
                    std::cout << val << " ";
                }
                std::cout << std::endl;
            }

            std::cout << std::endl;

            ts = testSMM(A, B, NUMBER_OF_RUNS);
            testPMM(A, B, NUMBER_OF_RUNS, p, ts);
            test1DMM(A, B, NUMBER_OF_RUNS, p, ts);
            test2DMM(A, B, NUMBER_OF_RUNS, p, ts);
        }
    }
    
    // Matrices of size m x n & n x q
    for(int p = 4; p <= 8; p += 4){
        for(int m = 1; m <= 10; m++){
            for(int n = 1; n <= 10; n++){
                for(int q = 10; q >= 1; q--){
                    std::cout << "m x n : n x q = ";
                    std::cout << m << " x " << n << " : " <<  n << " x " << q << std::endl;
                    std::cout << "Number of processors: " << p << std::endl;
                    std::cout << std::endl;

                    std::vector<std::vector<int>> A = generateMatrix(m, n);
                    std::vector<std::vector<int>> B = generateMatrix(n, q);
                    
                    std::cout << "A matrix:" << std::endl;
                    for (const auto &row : A)
                    {
                        for (int val : row)
                        {
                            std::cout << val << " ";
                        }
                        std::cout << std::endl;
                    }

                    std::cout << std::endl;

                    std::cout << "B matrix:" << std::endl;
                    for (const auto &row : B)
                    {
                        for (int val : row)
                        {
                            std::cout << val << " ";
                        }
                        std::cout << std::endl;
                    }

                    std::cout << std::endl;

                    ts = testSMM(A, B, NUMBER_OF_RUNS);
                    testPMM(A, B, NUMBER_OF_RUNS, p, ts);
                    test1DMM(A, B, NUMBER_OF_RUNS, p, ts);
                    test2DMM(A, B, NUMBER_OF_RUNS, p, ts);
                }
            }
        }
    }    
    

    //testMM(); // Run correctness tests

    // // OpenMP pragma directive to create a parallel region
    // #pragma omp parallel
    // {
    //     // Get the unique identifier of the current thread
    //     int ID = omp_get_thread_num();

    //     // Print a message indicating which thread is executing
    //     std::cout << "Hello from thread " << ID << std::endl;
    // }
    // // End of the parallel region

    return 0;
}

// Method implementations
std::vector<std::vector<int>> serialMM(const std::vector<std::vector<int>> &A, const std::vector<std::vector<int>> &B)
{
    /* Implementation */
    int m = A.size();    // number of rows in A
    int n = A[0].size(); // number of columns in A
    int q = B[0].size(); // number of columns in B

    // C matrix initialized with zeros
    std::vector<std::vector<int>> C(m, std::vector<int>(q, 0));

    // Perform matrix multiplication
    for (int i = 0; i < m; ++i)  // m == A.size()
    {
        for (int j = 0; j < q; ++j) // q == B[0].size()
        { 
            for (int k = 0; k < n; ++k)
            {                                 // n == B.size()
                C[i][j] += A[i][k] * B[k][j]; // perform the multiplication
            }
        }
    }

    return C;
}

std::vector<std::vector<int>> parallelMM(const std::vector<std::vector<int>> &A, const std::vector<std::vector<int>> &B, int P)
{
    int m = A.size();                                           // Number of rows in A
    int n = A[0].size();                                        // Number of columns in A / rows in B
    int q = B[0].size();                                        // Number of columns in B
    omp_set_num_threads(P);
    std::vector<std::vector<int>> C(m, std::vector<int>(q, 0)); // Resultant matrix

    // Parallelize the outer loop
    #pragma omp parallel for collapse(2) // collapse(2) is used to parallelize both the outer loops
    for (int i = 0; i < m; i++) // Number of rows in A
    { 
        for (int j = 0; j < q; j++)  // Number of columns in B
        {  
            int sum = 0; // Accumulator for the result of multiplication
            /**
             * Specifies that innermost loop should be parallelized.
             * Each thread gets its own copy of the sum variable to independently accumulate.
             * At the end of parallelization, all private copies are combined using '+'
            */
            #pragma omp parallel for reduction(+ : sum)
            for (int k = 0; k < n; k++) // Number of columns in A / rows in B
            { 
                sum += A[i][k] * B[k][j]; // Perform the multiplication
            }
            C[i][j] = sum; // Assign the result to the corresponding element of C
        }
    }

    return C;
}

/**
 * 1D Matrix Multiplication
 * Row-wise parallelization : each thread is responsible for operations of a row of A and column of B
*/
std::vector<std::vector<int>> oneDMM(const std::vector<std::vector<int>> &A, const std::vector<std::vector<int>> &B, int P) {
    int m = A.size();           // Number of rows in A
    int n = A[0].size();        // Number of columns in A (and number of rows in B)
    int q = B[0].size();        // Number of columns in B
    omp_set_num_threads(P);

    std::vector<std::vector<int>> C(m, std::vector<int>(q, 0)); // Resultant matrix
    /**
     * -Range of loops divide into chunks, with each chunk assigned to different thread.
     * -Each thread executes its assigned chunk independently -> this is okay because
     *  the results of one chunk (row of A, col of B) is not dependent on other chunks
     * -Parallelized over rows of resultant matrix
    */
    #pragma omp parallel for
    for (int i = 0; i < m; i++) // Rows of matrix A
    { 
        for (int j = 0; j < q; j++) // Columns of matrix B
        { 
            int sum = 0; // Accumulator for multiplication result
            for (int k = 0; k < n; k++) // Iterate over each element of the row in A and the column in B
            { 
                sum += A[i][k] * B[k][j]; // Multiply the elements and add to sum
            }
            C[i][j] = sum; // Assign the result to the corresponding element of C
        }
    }

    return C; // Return the resultant matrix C
}

/**
 * 2D Matrix Multiplication
 * Element-wise parallelization : each thread is responsible for operations of an element of C
*/
std::vector<std::vector<int>> twoDMM(const std::vector<std::vector<int>> &A, const std::vector<std::vector<int>> &B, int P) {
    int m = A.size();    // Number of rows in matrix A
    int n = A[0].size(); // Number of columns in matrix A / Number of rows in matrix B
    int q = B[0].size(); // Number of columns in matrix B
    std::vector<std::vector<int>> C(m, std::vector<int>(q, 0)); // Initialize the resultant matrix C with zeros
    omp_set_num_threads(P);

    /**
     * Collapse row/col loops into a single loop and divide equal-sized chunks workload among threads
     * for element-wise parallelization. Meaning, work is parallelized across elements of C
    */
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) // Iterate over each row of matrix A
    { 
        for (int j = 0; j < q; j++) // Iterate over each column of matrix B
        { 
            for (int k = 0; k < n; k++) // Perform the dot product
            { 
                /**
                 * Atomic Operation : ensure this op is completed as a whole without interference from other threads 
                 * Meaning, updates to shared memory are done atomically
                */
                #pragma omp atomic
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C; // Return the resultant matrix C
}

/**
 * Generate Square Matrix
*/
std::vector<std::vector<int>> generateSquareMatrix(int n) {
    std::vector<std::vector<int>> matrix(n, std::vector<int>(n));

    // Fill the matrix with random numbers between 1 and 1000
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = std::rand() % 1000 + 1;
        }
    }

    return matrix;
}

/**
 * Generate Matrix m x n
*/
std::vector<std::vector<int>> generateMatrix(int rows, int cols) {
    std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));

    // Fill the matrix with random numbers between 1 and 1000
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = std::rand() % 1000 + 1;
        }
    }

    return matrix;
}

double testSMM(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B, int runs){
    double sum = 0.0;

    // Measure time for serialMM
    for(int i = 0; i < runs; i++){
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<int>> C = serialMM(A, B);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_serial = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Time taken by serialMM: " << duration_serial.count() << " microseconds" << std::endl;
        sum += static_cast<double>(duration_serial.count());
        
        if(i == 4){
            std::cout << std::endl;
            // Display the C
            std::cout << "C matrix:" << std::endl;
            for (const auto &row : C)
            {
                for (int val : row)
                {
                    std::cout << val << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    double average = sum / runs;
    std::cout << "Average time taken by serialMM: " << average << " microseconds" << std::endl;
    std::cout << "=================================================" << std::endl;
    
    return average;
}

void testPMM(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B, int runs, int processors, double serial_time){
    double sum = 0.0;

    std::cout << std::endl;

    // Measure time for parallelMM
    for(int i = 0; i < runs; i++){
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<int>> C = parallelMM(A, B, processors);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_parallel = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Time taken by parallelMM: " << duration_parallel.count() << " microseconds" << std::endl;
        sum += static_cast<double>(duration_parallel.count());
        
        if(i == 4){
            std::cout << std::endl;
            // Display the C
            std::cout << "C matrix:" << std::endl;
            for (const auto &row : C)
            {
                for (int val : row)
                {
                    std::cout << val << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    double parallel_time = sum / runs;

    std::cout << "Average time taken by parallelMM: " << parallel_time << " microseconds" << std::endl;
    std::cout << "=================================================" << std::endl;

    double speedup = serial_time / parallel_time ;
    std::cout << "Speedup: " << speedup << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << "Cost: " << processors * parallel_time << " processor-mircroseconds" << std::endl;
    std::cout << "=================================================" << std::endl;
}

void test1DMM(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B, int runs, int processors, double serial_time){
    double sum = 0.0;

    std::cout << std::endl;

    // Measure time for 1DMM
    for(int i = 0; i < runs; i++){
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<int>> C = oneDMM(A, B, processors);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_1DMM = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Time taken by oneDMM: " << duration_1DMM.count() << " microseconds" << std::endl;
        sum += static_cast<double>(duration_1DMM.count());
        
        if(i == 4){
            std::cout << std::endl;
            // Display the C
            std::cout << "C matrix:" << std::endl;
            for (const auto &row : C)
            {
                for (int val : row)
                {
                    std::cout << val << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    double parallel_time = sum / runs;

    std::cout << "Average time taken by 1DMM: " << parallel_time << " microseconds" << std::endl;
    std::cout << "=================================================" << std::endl;

    double speedup = serial_time / parallel_time ;
    std::cout << "Speedup: " << speedup << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << "Cost: " << processors * parallel_time << " processor-mircroseconds" << std::endl;
    std::cout << "=================================================" << std::endl;
}

void test2DMM(std::vector<std::vector<int>> A, std::vector<std::vector<int>> B, int runs, int processors, double serial_time){
    double sum = 0.0;

    std::cout << std::endl;

    // Measure time for 2DMM
    for(int i = 0; i < runs; i++){
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<int>> C = twoDMM(A, B, processors);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_2DMM = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Time taken by 2DMM: " << duration_2DMM.count() << " microseconds" << std::endl;
        sum += static_cast<double>(duration_2DMM.count());
        
        if(i == 4){
            std::cout << std::endl;
            // Display the C
            std::cout << "C matrix:" << std::endl;
            for (const auto &row : C)
            {
                for (int val : row)
                {
                    std::cout << val << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    double parallel_time = sum / runs;

    std::cout << "Average time taken by 2DMM: " << parallel_time << " microseconds" << std::endl;
    std::cout << "=================================================" << std::endl;

    double speedup = serial_time / parallel_time ;
    std::cout << "Speedup: " << speedup << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << "Cost: " << processors * parallel_time << " processor-mircroseconds" << std::endl;
    std::cout << "=================================================" << std::endl;

    std::cout << std::endl;
    std::cout << std::endl;
}

// This is a correctness test, not a performance test
void testMM()
{
   // Call MM functions with test cases

    // Calling the serialMM function to run on example matrices A and B
    std::vector<std::vector<int>> A = {{1, 2, 3}, {4, 5, 6}};
    std::vector<std::vector<int>> B = {{7, 8}, {9, 10}, {11, 12}};

    // Perform matrix multiplication
    // std::vector<std::vector<int>> C = serialMM(A, B, DEFAULT_NUM_THREADS);
    // std::vector<std::vector<int>> C = parallelMM(A, B, DEFAULT_NUM_THREADS);
    // std::vector<std::vector<int>> C = oneDMM(A, B, DEFAULT_NUM_THREADS);
    std::vector<std::vector<int>> C = twoDMM(A, B, DEFAULT_NUM_THREADS);

    // Display the C
    std::cout << "C matrix:" << std::endl;
    for (const auto &row : C)
    {
        for (int val : row)
        {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    // Compare Cs to expected outcomes and report any discrepancies
    std::vector<std::vector<int>> expected = {{58, 64}, {139, 154}};
    if (C == expected)
    {
        std::cout << "Test passed!" << std::endl;
    }
    else
    {
        std::cout << "Test failed!" << std::endl;
    }
}