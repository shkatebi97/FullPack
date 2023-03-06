#include "profiler.h"

using namespace std;

namespace SelfProfiler{

Profiler* _main_profiler = nullptr;

void ProfilerTimer::start(){
    if(_parent == nullptr) 
        throw string("No Parent Assigned"); 
    if(_finished)
        throw string("Already Ended");
    if(_name == "")
        throw string("Has No Name");
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &_start);
    _started  = true;
    _finished = false;
}
void ProfilerTimer::end(){
    if(_parent == nullptr)
        throw string("No Parent Assigned");
    if(!_started)
        throw string("Did not started");
    if(_name == "")
        throw string("Has No Name");
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &_end); 
    _started  = false; 
    _finished = true; 
    ((Profiler*)_parent)->TimerDone(_name, calc_time());
}
void ProfilerTimer::set_parent(void* parent){
    _parent = parent;
}

void Profiler::TimerDone(string name, double long consumed_time){
    pair<string, long double> iPair;
    iPair.first = name;
    iPair.second = consumed_time;
    _times.push_back(iPair);
}
ProfilerTimer* Profiler::NewTimer(std::string name){
    ProfilerTimer* timer = new ProfilerTimer(0);
    timer->set_parent((void*)this);
    timer->operator()(name);
    pair<long, ProfilerTimer*> oPair;
    oPair.first = last_id;
    oPair.second = timer;
    _timers.push_back(oPair);
    return timer;
}
long double Profiler::queryTotalTime(std::string name){
    long double totalTime = -0.0001;
    for(auto i : _times)
        if (i.first == name || i.first.substr(0, i.first.find(":")) == name)
            if(totalTime < 0)
                totalTime = i.second;
            else
                totalTime += i.second;
    return totalTime;
}
std::vector<std::string> Profiler::getAllNames(){
    std::vector<std::string> t;
    for(auto i : _times)
        t.push_back(i.first);
    std::set<std::string> _set(t.begin(), t.end());
    t.clear();
    std::vector<std::string> _vec(_set.begin(), _set.end());
    return _vec;
}

Profiler* get_main_profiler(){
    if (_main_profiler == nullptr)
        _main_profiler = new Profiler;
    return _main_profiler;
}

}
