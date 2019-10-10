#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/ThreadPool"
#include "unsupported/Eigen/CXX11/Tensor"
#include<iostream>
#include <chrono> 
using namespace std::chrono; 

int tasks = 1000000;
int kthreads = 16;


static void test_creation(bool allow_spinning)
{
    auto time_start = high_resolution_clock::now(); 
    for (int i = 0; i < tasks; i++) {
        Eigen::ThreadPool pool(kthreads, allow_spinning);        
    }
    auto time_end = high_resolution_clock::now(); 
    auto t = duration_cast<microseconds>(time_end - time_start).count() * 1e-6; 
    double time_avg = t/tasks;
    std::cout<<" Total time  : "<<t<<" secs \n";
    std::cout<<" Average time: "<<time_avg<<" secs \n";
}


static void test_parallelism(bool allow_spinning)
{
    Eigen::ThreadPool tp(kthreads, allow_spinning);

    auto time_start = high_resolution_clock::now(); 

    for (int iter = 0; iter < 10; ++iter) {
        std::atomic<int> sum(0);
        for (int i = 0; i < tasks; ++i) {
            tp.Schedule([&]() {
            sum++;
            });
        }
    }
    auto time_end = high_resolution_clock::now(); 
    auto t = duration_cast<microseconds>(time_end - time_start).count() * 1e-6; 
    double time_avg = t/10;
    std::cout<<" Total time  : "<<t<<" secs \n";
    std::cout<<" Average time: "<<time_avg<<" secs \n";
}

int main(int argc, char *argv[])
{
    Eigen::initParallel();
    kthreads = atoi(argv[1]);
    tasks = atoi(argv[2]);

    printf (" =============== Eigen ================\n");
    printf (" Creating %i threads\n", kthreads);
    printf (" Submitting %i tasks\n", tasks);
    test_parallelism(true);  
    return 0;
}
