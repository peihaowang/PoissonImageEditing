
#ifndef STOP_WATCH_H
#define STOP_WATCH_H

#include <iostream>
#include <time.h>

class StopWatch
{
private:
    clock_t         m_startTime;
    clock_t         m_lastTime;
public:
    StopWatch() : m_startTime(std::clock()), m_lastTime(m_startTime) { return; }

    double escapeTime() const { return (double)(std::clock() - m_lastTime) / CLOCKS_PER_SEC; }
    double totalTime() const { return (double)(std::clock() - m_startTime) / CLOCKS_PER_SEC; }

    void tick(const char* title)
    {
        std::cout << title << ": " << escapeTime() << "s" << std::endl;
        m_lastTime = std::clock();
    }

};

#endif // STOP_WATCH_H
