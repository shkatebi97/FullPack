#ifndef PROFILER_H
#define PROFILER_H

#include <iostream>
#include <algorithm>
#include <vector>
#include <set>
#include <time.h>
#include <unistd.h>

namespace SelfProfiler{

class Profiler;

class ProfilerTimer{
    void* _parent;
    const long _id;
    std::string _name;
    struct timespec _start, _end;
    bool _started, _finished;
    long double calculate_time_diff_seconds(timespec start, timespec end){
        long double tstart_ext_ld, tend_ext_ld, tdiff_ext_ld;
        tstart_ext_ld = (long double)start.tv_sec + start.tv_nsec/1.0e+9;
        tend_ext_ld = (long double)end.tv_sec + end.tv_nsec/1.0e+9;
        tdiff_ext_ld = (long double)(tend_ext_ld - tstart_ext_ld);
        return tdiff_ext_ld;
    }
    long double calc_time(){return this->calculate_time_diff_seconds(_start, _end);}
public:
    ProfilerTimer(long id):
        _id(id), 
        _name(""), 
        _start({0,0}), 
        _end({0,0}), 
        _started(false), 
        _finished(false), 
        _parent(nullptr){}
    ~ProfilerTimer(){}
    void operator()(std::string name){_name = name;}
    void start();
    void end();
    void set_parent(void* parent);
};

class Profiler{
private:
    long last_id;
    std::vector<std::pair<std::string, long double>> _times;
    std::vector<std::pair<long, ProfilerTimer*>> _timers;
    void TimerDone(std::string name, double long consumed_time);
public:
    Profiler():last_id(0){}
    ~Profiler(){}
    ProfilerTimer* NewTimer(std::string name);
    long double queryTotalTime(std::string name);
    std::vector<std::string> getAllNames();
    friend class ProfilerTimer;
};

extern Profiler* _main_profiler;

Profiler* get_main_profiler();
}

#endif