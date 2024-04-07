# 5350GroupProj1

Implementing MM Algorithms
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
1) Serial Algorithm (MM-ser):
Start with a straightforward implementation using three nested for-loops to compute the result matrix C from matrices A and B.

2) Simple Parallel Algorithm (MM-Par):
Use #pragma omp parallel for to parallelize the outermost loop of your serial implementation. This approach automatically distributes iterations of the loop across the available threads.

3) 1D Parallel Algorithm (MM-1D):
Assign each row of the result matrix C to a different thread. You might need to partition the outer loop so each thread works on a different set of rows.

4) 2D Parallel Algorithm (MM-2D):
This is more complex, as you'll divide the matrices into sub-blocks and assign each sub-block to a thread. Consider the matrix dimensions and ensure they are divisible by the square root of P (number of threads) for simplicity.


Designing the Performance Study
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
- Experiment Design: Run your implementations with varying sizes of matrices (m, n, q) and different numbers of threads (P). Use matrices where dimensions are multiples of common thread counts (4, 16, 64) to simplify the 2D partitioning.
- Metrics: Measure execution time for each setup. Calculate speedup as T_serial / T_parallel, where T_serial is the execution time of the serial version and T_parallel is the time for the parallel version. The cost is calculated as P * T_parallel.
- Analysis: Look for trends in how matrix dimensions and the number of threads affect performance. Consider the overhead of parallelization and how effectively the workload is distributed among threads.


Analyzing Results
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
- Graphs: Plot speedup and cost against the number of threads for fixed matrix sizes, and vice versa. Look for points of diminishing returns or where overhead dominates.
- Tables: Summarize execution times, speedup, and cost for key configurations.
- Discussion: Analyze why certain configurations work better, considering factors like workload distribution, overhead, and hardware limitations (e.g., the number of physical cores).


Tools and Environment
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
- OpenMP Compilation: Use the -fopenmp flag with GCC to enable OpenMP.
- Cal Poly HPC Center: Familiarize yourself with submitting jobs and accessing resources. Your jobs will likely be batch submitted, so ensure your scripts correctly set the number of threads and capture all necessary output.


Deliverables
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
- Ensure your PowerPoint slides are clear and concise, presenting your methodology, results, and analyses effectively. Include pseudocode for clarity.
- Organize your source code for readability and include comments explaining critical sections, especially where parallelization is applied.
- Test your implementations thoroughly to ensure correctness before running performance experiments.


Responsibilties (All necessary information provided above)
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
- Implement the algorithms (serial, par, 1d, 2d)
- Figure out how Cal Poly HPC Center handles jobs (batch submission), scripts, number of threads and output capture. Basically, we need to know how to integrate with our implementations
- Design, implement, analyze, and present performance studies