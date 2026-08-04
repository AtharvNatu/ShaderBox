#ifndef PERFORMANCE_STATS_HPP
#define PERFORMANCE_STATS_HPP

#define NOMINMAX
#include <Windows.h>
#include <nvml.h>
#include <Pdh.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

#pragma comment(lib, "nvml.lib")
#pragma comment(lib, "pdh.lib")

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
        float averageFPS = 0.0f;

        float accumulatedTime = 0.0f;
        uint32_t accumulatedFrames = 0;

        static constexpr float smoothingTimeConstant = 0.5f;

        static constexpr size_t HISTORY_SIZE = 120;
        static constexpr float historySampleInteval = 1.0f / 60.0f;

        float historySampleAccumulator = 0.0f;
        std::vector<float> fpsHistory;
        std::vector<float> frameTimeHistory;

        bool statsPrimed = false;

    public:
        void update()
        {
            deltaTime = frameTimer.tick();
            frameTime = deltaTime * 1000.0f;

            if (!statsPrimed)
            {
                currentFPS = 1.0f / deltaTime;
                smoothedFrameTime = frameTime;
                statsPrimed = true;
            }
            else
            {
                float alpha = 1.0f - expf(-deltaTime / smoothingTimeConstant);
                currentFPS += alpha * ((1.0f / deltaTime) - currentFPS);
                smoothedFrameTime += alpha * (frameTime - smoothedFrameTime);
            }

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
            accumulatedTime = 0.0f;
            accumulatedFrames = 0;
            historySampleAccumulator = 0;
            statsPrimed = false;
        }

        float getDeltaTime() const
        {
            return deltaTime;
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
            if (fpsHistory.empty())
                return 0.0f;

            return *std::min_element(fpsHistory.begin(), fpsHistory.end());
        }

        float getMaximumFPS() const 
        { 
            if (fpsHistory.empty())
                return 0.0f;

            return *std::max_element(fpsHistory.begin(), fpsHistory.end());
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
        PDH_HQUERY pdhQuery = nullptr;
        PDH_HCOUNTER cpuUtilCounter = nullptr;
        MEMORYSTATUSEX memoryStatusEx;
        bool bQuerySucceeded = false;

        float cpuUsagePercentage = 0.0f;
        float memoryUsagePercentage = 0.0f;
        float memoryUsedGB = 0.0f;
        float memoryTotalGB = 0.0f;

        ULONGLONG lastRefreshTick = 0;
        
        // Nvidia GPU Related
        nvmlReturn_t nvmlRet;
        nvmlDevice_t nvmlDevice;
        nvmlUtilization_t nvmlUtilization;
        nvmlMemory_t nvmlMemory;
        bool bDeviceFound = false;
        unsigned int deviceCount;
        std::string vkDeviceUUID;
        unsigned int gpuUsagePercentage = 0;
        float vramUsedGB = 0.0f;
        float vramTotalGB = 0.0f;

        FILE *logFile = NULL;

        static constexpr ULONGLONG uRefreshIntervalMs = 300;

        //! Prevent shallow copy
        SystemStats(const SystemStats&) = delete;
        SystemStats& operator = (const SystemStats&) = delete;

    public:
        SystemStats(FILE** pLogFile, std::string _vkDeviceUUID)
        {
            // Code
            logFile = *pLogFile;
            vkDeviceUUID = _vkDeviceUUID;

            //* CPU Utilization
            if (PdhOpenQuery(nullptr, 0, &pdhQuery) != ERROR_SUCCESS)
            {
                fprintf(logFile, "%s() => PdhOpenQuery() Failed !!!\n", __func__);
                bQuerySucceeded = false;
                return;
            }

            if (PdhAddEnglishCounterW(pdhQuery, L"\\Processor Information(_Total)\\% Processor Utility", 0, &cpuUtilCounter) != ERROR_SUCCESS)
            {
                fprintf(logFile, "%s() => PdhAddEnglishCounter() Failed !!!\n", __func__);
                bQuerySucceeded = false;
                return;
            }

            // Prime the first sample
            if (PdhCollectQueryData(pdhQuery) == ERROR_SUCCESS)
                bQuerySucceeded = true;

            //* Memory Query
            memset((void*)&memoryStatusEx, 0, sizeof(MEMORYSTATUSEX));
            memoryStatusEx.dwLength = sizeof(memoryStatusEx);
            GlobalMemoryStatusEx(&memoryStatusEx);
            memoryTotalGB = (float)(memoryStatusEx.ullTotalPhys / (1024.0f * 1024.0f * 1024.0f));

            //* Nvidia GPU Utilization
            nvmlRet = nvmlInit_v2();
            if (nvmlRet != NVML_SUCCESS) 
            {
                fprintf(logFile, "%s() => nvmlInit_v2() Failed !!!\n", __func__);
                return;
            }
            else
                fprintf(logFile, "%s() => nvmlInit_v2() Succeeded\n", __func__);

            memset((void*)&nvmlDevice, 0, sizeof(nvmlDevice_t));

            nvmlRet = nvmlDeviceGetCount_v2(&deviceCount);
            if (nvmlRet != NVML_SUCCESS) 
            {
                fprintf(logFile, "%s() => nvmlDeviceGetCount_v2() Failed !!!\n", __func__);
                return;
            }
            else if (deviceCount == 0) 
            {
                fprintf(logFile, "%s() => nvmlDeviceGetCount_v2() Returned 0 Devices !!!\n", __func__);
                return;
            }

            nvmlRet = nvmlDeviceGetHandleByUUID(vkDeviceUUID.c_str(), &nvmlDevice);
            if (nvmlRet != NVML_SUCCESS) 
            {
                fprintf(logFile, "%s() => nvmlDeviceGetHandleByUUID() Failed : %d !!!\n", __func__, nvmlRet);
                return;
            }

            memset((void*)&nvmlUtilization, 0, sizeof(nvmlUtilization_t));
            memset((void*)&nvmlMemory, 0, sizeof(nvmlMemory_t));

            bDeviceFound = true;

            // Total VRAM - Checked only once
            if (bDeviceFound)
            {
                if (nvmlDeviceGetMemoryInfo(nvmlDevice, &nvmlMemory) == NVML_SUCCESS)
                    vramTotalGB = static_cast<float>(nvmlMemory.total) / (1024.0f * 1024.0f * 1024.0f);
            }

            lastRefreshTick = GetTickCount64();
            
        }

        void update()
        {
            // Code
            ULONGLONG now = GetTickCount64();
            if (now - lastRefreshTick >= uRefreshIntervalMs)
            {
                lastRefreshTick = now;

                // CPU Usage
                if (bQuerySucceeded)
                {
                    if (PdhCollectQueryData(pdhQuery) == ERROR_SUCCESS)
                    {
                        PDH_FMT_COUNTERVALUE pdhCounterValue;
                        if (PdhGetFormattedCounterValue(cpuUtilCounter, PDH_FMT_DOUBLE, nullptr, &pdhCounterValue) == ERROR_SUCCESS)
                        {
                            if (pdhCounterValue.CStatus == ERROR_SUCCESS)
                            {
                                // Clamp value
                                if (pdhCounterValue.doubleValue > 100.0)
                                    cpuUsagePercentage = 100.0;
                                else
                                    cpuUsagePercentage = pdhCounterValue.doubleValue;
                            }
                            
                        }
                    }
                    else
                        cpuUsagePercentage = -1.0;
                }

                // Memory Usage
                GlobalMemoryStatusEx(&memoryStatusEx);
                memoryUsagePercentage = (float)memoryStatusEx.dwMemoryLoad;
                memoryUsedGB = memoryTotalGB - (float)(memoryStatusEx.ullAvailPhys / (1024.0f * 1024.0f * 1024.0f));

                // GPU Usage
                if (bDeviceFound)
                {  
                    if (nvmlDeviceGetUtilizationRates(nvmlDevice, &nvmlUtilization) == NVML_SUCCESS)
                        gpuUsagePercentage = nvmlUtilization.gpu;

                    if (nvmlDeviceGetMemoryInfo(nvmlDevice, &nvmlMemory) == NVML_SUCCESS)
                        vramUsedGB = static_cast<float>(nvmlMemory.used) / (1024.0f * 1024.0f * 1024.0f);
                }
            
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

        unsigned int getGPUUsage() const
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
            nvmlRet = nvmlShutdown();
            if (nvmlRet != NVML_SUCCESS) 
                    fprintf(logFile, "%s() => nvmlShutdown() Failed !!!\n", __func__);

            if (pdhQuery)
            {
                // Invalidates cpuUtilCounter
                PdhCloseQuery(pdhQuery);
                pdhQuery = nullptr;
                cpuUtilCounter = nullptr;
            }
        }
};

#endif
