#include <folly/executors/CPUThreadPoolExecutor.h>
#include <chrono> 
#include <iostream>
using namespace std::chrono; 
using namespace std;
using namespace folly;

int kthreads = 4;
int tasks = 10;

void test_creation(){

    auto time_start = high_resolution_clock::now(); 
    for(int i=0; i<tasks; i++){
        CPUThreadPoolExecutor tp(kthreads);
    }
    auto time_end = high_resolution_clock::now(); 
    auto t = duration_cast<microseconds>(time_end - time_start).count() * 1e-6; 
    double time_avg = t/tasks;
    printf(" Total time  : %f secs\n", t);
    printf(" Average time: %f secs\n", time_avg);
}


void test_parallelism(){

    auto time_start = high_resolution_clock::now(); 

    for (int iter = 0; iter < 10; ++iter) {
        CPUThreadPoolExecutor tp(kthreads);
        std::atomic<int> sum(0);
        for (int i = 0; i < tasks; ++i) {
            tp.add([&]() {
            sum++;
            });
        }
        //if(int(sum) != tasks)
        //  std::cout<<"!!!"<<sum<<"\n";
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

    printf (" =============== folly ================\n");
    printf (" Creating %i threads\n", kthreads);
    test_parallelism();

}
