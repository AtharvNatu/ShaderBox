#ifndef PERFORMANCE_STATS_HPP
#define PERFORMANCE_STATS_HPP

#define NOMINMAX
#include <Windows.h>
#include <algorithm>

class FrameTimer
{
    private:
        LARGE_INTEGER frequency;
        LARGE_INTEGER lastCounter;
        float deltaTime = 0.0f;

    public:
        FrameTimer()
        {
            QueryPerformanceFrequency(&frequency);
            QueryPerformanceCounter(&lastCounter);
        }

        float tick()
        {
            LARGE_INTEGER currentCounter;
            QueryPerformanceCounter(&currentCounter);

            deltaTime = static_cast<float>(currentCounter.QuadPart - lastCounter.QuadPart) / static_cast<float>(frequency.QuadPart);
            lastCounter = currentCounter;
            return deltaTime;
        }
};

class PerformanceStats
{
    private:
        FrameTimer frameTimer;
        float deltaTime;

        float frameTime = 0.0f;
        float currentFPS = 0.0f;
        float minFPS = FLT_MAX;
        float maxFPS = 0.0f;
        float averageFPS = 0.0f;

        float accumulatedTime = 0.0f;
        uint32_t accumulatedFrames = 0;

    public:
        void update()
        {
            deltaTime = frameTimer.tick();
            frameTime = deltaTime * 1000.0f;

            currentFPS = currentFPS * 0.9f + (1.0f / deltaTime) * 0.1f;

            minFPS = std::min(minFPS, currentFPS);
            maxFPS = std::max(maxFPS, currentFPS);

            accumulatedTime += deltaTime;
            accumulatedFrames++;

            if (accumulatedTime >= 1.0f)
            {
                averageFPS = accumulatedFrames / accumulatedTime;
                accumulatedTime -= 1.0f;
                accumulatedFrames = 0;
            }
        }

        void reset()
        {
            currentFPS = 0.0f;
            averageFPS = 0.0f;
            frameTime = 0.0f;
            minFPS = FLT_MAX;
            maxFPS = 0.0f;
            accumulatedTime = 0.0f;
            accumulatedFrames = 0;
        }

        float getFrameTime() const 
        { 
            return frameTime; 
        }

        float getFPS() const 
        { 
            return currentFPS; 
        }

        float getMinimumFPS() const 
        { 
            return (minFPS == FLT_MAX) ? 0.0f : minFPS; 
        }

        float getMaximumFPS() const 
        { 
            return maxFPS; 
        }

        float getAverageFPS() const 
        { 
            return averageFPS; 
        }
};


#endif
