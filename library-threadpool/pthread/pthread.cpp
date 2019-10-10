#include <iostream>
#include "threadpool.h"
#include <chrono> 
using namespace std::chrono; 

int kthreads = 4;
int tasks = 10;

void test_creation(){
    auto time_start = high_resolution_clock::now(); 
    for(int i=0; i<tasks; i++)
        ThreadPool tp(kthreads, 0);
    auto time_end = high_resolution_clock::now(); 
    auto t = duration_cast<microseconds>(time_end - time_start).count() * 1e-6; 
    double time_avg = t/tasks;
    std::cout<<" Total time  : "<<t<<" secs \n";
    std::cout<<" Average time: "<<time_avg<<" secs \n";
}


void test_parallelism(){
    ThreadPool tp(kthreads, 0);
    auto time_start = high_resolution_clock::now(); 

    for (int iter = 0; iter < 10; ++iter) {
        std::atomic<int> sum(0);
        for (int i = 0; i < tasks; ++i) {
            tp.run([&]() {
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
    kthreads = atoi(argv[1]);
    tasks = atoi(argv[2]);

    printf (" =============== pthread ================\n");
    printf (" Creating %i threads\n", kthreads);
    printf (" Submitting %i tasks\n", tasks);
    test_parallelism();

  return 0;
}

