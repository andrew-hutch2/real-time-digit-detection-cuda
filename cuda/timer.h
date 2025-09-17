#include <time.h>
#include <map>
#include <string>
using namespace std;

// Simple timer class for automatic timing
class Timer {
    private:
        struct timespec start_time;
        string name;
        static map<string, double> timings;
        
    public:
        Timer(const string& timer_name) : name(timer_name) {
            clock_gettime(CLOCK_MONOTONIC, &start_time);
        }
        
        ~Timer() {
            struct timespec end_time;
            clock_gettime(CLOCK_MONOTONIC, &end_time);
            double elapsed = (end_time.tv_sec - start_time.tv_sec) + 
                            (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
            timings[name] += elapsed;
        }
        
        static void reset() {
            timings.clear();
        }
        
        static void print_timings() {
            printf("\n=== CUDA GPU IMPLEMENTATION TIMING BREAKDOWN ===\n");
            
            double total_time = 0.0;
            for (const auto& pair : timings) {
                total_time += pair.second;
            }
            
            printf("Total training time: %.1f seconds\n\n", total_time);
            printf("Detailed Breakdown:\n");
            
            for (const auto& pair : timings) {
                printf("  %-15s: %6.3fs (%5.1f%%)\n", 
                       pair.first.c_str(), pair.second, 100.0 * pair.second / total_time);
            }
        }
    };