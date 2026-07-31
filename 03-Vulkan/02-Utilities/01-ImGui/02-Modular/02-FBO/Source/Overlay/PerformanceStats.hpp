#ifndef PERFORMANCE_STATS_HPP
#define PERFORMANCE_STATS_HPP

#pragma comment(lib, "pdh.lib")

#define NOMINMAX
#include <Windows.h>
#include <Pdh.h>
#include <algorithm>

class FrameTimer
{
    private:
        LARGE_INTEGER frequency;
        LARGE_INTEGER lastCounter;
        float deltaTime = 0.0f;
        bool firstTick = true;

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

            if (firstTick)
            {
                firstTick = false;
                lastCounter = currentCounter;
                deltaTime = 1.0f / 60.0f;
                return deltaTime;
            }

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
        float smoothedFrameTime = 0.0f;
        float currentFPS = 0.0f;
        float minFPS = FLT_MAX;
        float maxFPS = 0.0f;
        float averageFPS = 0.0f;

        float accumulatedTime = 0.0f;
        uint32_t accumulatedFrames = 0;

        static constexpr float smoothingTimeConstant = 0.5f;

        static constexpr size_t HISTORY_SIZE = 120;
        static constexpr float historySampleInteval = 1.0f / 60.0f;

        float historySampleAccumulator = 0.0f;
        std::vector<float> fpsHistory;
        std::vector<float> frameTimeHistory;

    public:
        void update()
        {
            deltaTime = frameTimer.tick();
            frameTime = deltaTime * 1000.0f;
            
            float alpha = 1.0f - expf(-deltaTime / smoothingTimeConstant);
            currentFPS += alpha * ((1.0f / deltaTime) - currentFPS);
            smoothedFrameTime += alpha * (frameTime - smoothedFrameTime);

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

            historySampleAccumulator += deltaTime;
            if (historySampleAccumulator >= historySampleInteval)
            {
                historySampleAccumulator -= historySampleInteval;
                
                fpsHistory.push_back(currentFPS);
                if (fpsHistory.size() > HISTORY_SIZE)
                    fpsHistory.erase(fpsHistory.begin());

                frameTimeHistory.push_back(smoothedFrameTime);
                if (frameTimeHistory.size() > HISTORY_SIZE)
                    frameTimeHistory.erase(frameTimeHistory.begin());
            }
        }

        void reset()
        {
            currentFPS = 0.0f;
            averageFPS = 0.0f;
            frameTime = 0.0f;
            smoothedFrameTime = 0.0f;
            minFPS = FLT_MAX;
            maxFPS = 0.0f;
            accumulatedTime = 0.0f;
            accumulatedFrames = 0;
            historySampleAccumulator = 0;
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

        const std::vector<float>& getFPSHistory() const
        {
            return fpsHistory;
        }

        const std::vector<float>& getFrameTimeHistory() const
        {
            return frameTimeHistory;
        }
};

class SystemStats
{
    private:

        // CPU and Memory Related
        LARGE_INTEGER cpuCounterFrequency{};
        LARGE_INTEGER previousCpuCounter{};

        ULONGLONG previousProcessKernel = 0;
        ULONGLONG previousProcessUser = 0;

        DWORD logicalProcessorCount = 1;
        bool cpuPrimed = false;

        // GPU Related (Using PDH (Performance Data Helper) For Cross-Vendor GPU Support)
        PDH_HQUERY pdhQuery = nullptr;
        PDH_HCOUNTER gpuUtilCounter = nullptr;
        PDH_HCOUNTER vramUsedCounter = nullptr;

        float cpuUsagePercentage = 0.0f;
        float memoryUsagePercentage = 0.0f;
        float memoryUsedGB = 0.0f;
        float memoryTotalGB = 0.0f;
        float gpuUsagePercentage = 0.0f;
        float vramUsedGB = 0.0f;
        float vramTotalGB = 0.0f;

        ULONGLONG lastCpuRefreshTick = 0;
        ULONGLONG lastMemoryRefreshTick = 0;

    public:
        SystemStats()
        {
            // Code
            logicalProcessorCount = std::max<DWORD>(1,GetActiveProcessorCount(ALL_PROCESSOR_GROUPS));

            QueryPerformanceFrequency(&cpuCounterFrequency);
            QueryPerformanceCounter(&previousCpuCounter);

            cpuPrimed = getCurrentProcessCpuTimes(previousProcessKernel, previousProcessUser);

            PdhOpenQuery(NULL, 0, &pdhQuery);
            PdhAddEnglishCounterA(pdhQuery, "\\GPU Engine(*)\\Utilization Percentage", 0, &gpuUtilCounter);
            PdhAddEnglishCounterA(pdhQuery, "\\GPU Process Memory(*)\\Local Usage", 0, &vramUsedCounter);
            PdhCollectQueryData(pdhQuery);
        }

        static ULONGLONG fileTimeToUInt64(const FILETIME& time)
        {
            // Code
            ULARGE_INTEGER value{};
            value.LowPart = time.dwLowDateTime;
            value.HighPart = time.dwHighDateTime;
            return value.QuadPart;
        }

        static bool getCurrentProcessCpuTimes(ULONGLONG& kernelTime, ULONGLONG& userTime)
        {
            // Code
            FILETIME creation{};
            FILETIME exit{};
            FILETIME kernel{};
            FILETIME user{};

            if (!GetProcessTimes(
                GetCurrentProcess(),
                &creation,
                &exit,
                &kernel,
                &user
            ))
            {
                return false;
            }

            kernelTime = fileTimeToUInt64(kernel);
            userTime = fileTimeToUInt64(user);
            return true;
        }

        void update()
        {
            // Code

            ULONGLONG now = GetTickCount64();

            // Memory Usage
            if (now - lastMemoryRefreshTick >= 250)
            {
                lastMemoryRefreshTick = now;

                MEMORYSTATUSEX memoryStatusEx;
                memset((void*)&memoryStatusEx, 0, sizeof(MEMORYSTATUSEX));
                memoryStatusEx.dwLength = sizeof(memoryStatusEx);
                
                GlobalMemoryStatusEx(&memoryStatusEx);

                memoryUsagePercentage = (float)memoryStatusEx.dwMemoryLoad;
                memoryTotalGB = (float)(memoryStatusEx.ullTotalPhys / (1024.0f * 1024.0f * 1024.0f));
                memoryUsedGB = memoryTotalGB - (float)(memoryStatusEx.ullAvailPhys / (1024.0f * 1024.0f * 1024.0f));
            }
            
            // CPU Usage
            if (now - lastCpuRefreshTick >= 1000)
            {
                lastCpuRefreshTick = now;

                LARGE_INTEGER currentCounter{};
                QueryPerformanceCounter(&currentCounter);

                ULONGLONG currentKernel = 0;
                ULONGLONG currentUser = 0;

                if (getCurrentProcessCpuTimes(currentKernel, currentUser))
                {
                    if (cpuPrimed)
                    {
                        const double elapsedSeconds =
                            static_cast<double>(
                                currentCounter.QuadPart -
                                previousCpuCounter.QuadPart) /
                            static_cast<double>(cpuCounterFrequency.QuadPart);

                        const ULONGLONG kernelDelta =
                            currentKernel - previousProcessKernel;

                        const ULONGLONG userDelta =
                            currentUser - previousProcessUser;

                        // FILETIME CPU durations are in 100-nanosecond units.
                        const double processCpuSeconds =
                            static_cast<double>(kernelDelta + userDelta) *
                            1.0e-7;

                        if (elapsedSeconds > 0.0)
                        {
                            const double percentage =
                                100.0 * processCpuSeconds /
                                (elapsedSeconds *
                                static_cast<double>(logicalProcessorCount));

                            cpuUsagePercentage =
                                static_cast<float>(
                                    std::clamp(percentage, 0.0, 100.0));
                        }
                    }

                    cpuPrimed = true;
                    previousCpuCounter = currentCounter;
                    previousProcessKernel = currentKernel;
                    previousProcessUser = currentUser;
                }
            }

            // GPU Usage
            PdhCollectQueryData(pdhQuery);

            DWORD bufferSize = 0, itemCount = 0;
            PdhGetFormattedCounterArrayA(gpuUtilCounter, PDH_FMT_DOUBLE, &bufferSize, &itemCount, nullptr);
            if (bufferSize > 0)
            {
                std::vector<BYTE> buffer(bufferSize);

                PDH_FMT_COUNTERVALUE_ITEM_A* items = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_A*>(buffer.data());
                PdhGetFormattedCounterArrayA(gpuUtilCounter, PDH_FMT_DOUBLE, &bufferSize, &itemCount, items);
                
                double util3D = 0.0;
                for (DWORD i = 0; i < itemCount; i++)
                {
                    if (items[i].FmtValue.CStatus == ERROR_SUCCESS 
                        && strstr(items[i].szName, "engtype_3D"))
                        util3D += items[i].FmtValue.doubleValue;
                }

                gpuUsagePercentage = static_cast<float>(std::min(util3D, 100.0));
            }

            // VRAM Usage
            bufferSize = 0, itemCount = 0;
            PdhGetFormattedCounterArrayA(vramUsedCounter, PDH_FMT_LARGE, &bufferSize, &itemCount, nullptr);
            if (bufferSize > 0)
            {
                std::vector<BYTE> buffer(bufferSize);

                PDH_FMT_COUNTERVALUE_ITEM_A* items = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_A*>(buffer.data());
                PdhGetFormattedCounterArrayA(vramUsedCounter, PDH_FMT_LARGE, &bufferSize, &itemCount, items);
                
                char pidTag[32];
                sprintf_s(pidTag, "pid_%lu_", GetCurrentProcessId());

                LONGLONG total = 0;
                for (DWORD i = 0; i < itemCount; i++)
                {
                    if (items[i].FmtValue.CStatus == ERROR_SUCCESS 
                        && strstr(items[i].szName, pidTag))
                        total += items[i].FmtValue.largeValue;
                }

                vramUsedGB = static_cast<float>(total / (1024.0f * 1024.0f * 1024.0f));
            }

        }

        float getCPUUsage() const
        {
            return cpuUsagePercentage;
        }
        
        float getMemoryUsage() const
        {
            return memoryUsagePercentage;
        }

        float getMemoryUsed() const
        {
            return memoryUsedGB;
        }

        float getMemoryTotal() const
        {
            return memoryTotalGB;
        }

        float getGPUUsage() const
        {
            return gpuUsagePercentage;
        }

        float getVRAMUsed() const
        {
            return vramUsedGB;
        }

        float getVRAMTotal() const
        {
            return vramTotalGB;
        }

        ~SystemStats()
        {
            if (pdhQuery)
            {
                PdhCloseQuery(pdhQuery);
                pdhQuery = nullptr;
            }
        }
};

#endif
