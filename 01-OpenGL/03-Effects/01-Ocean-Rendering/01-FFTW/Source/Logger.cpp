#include "Logger.hpp"

// Class Instance
Logger *Logger::_logger = nullptr;

Logger::Logger(const std::string file)
{
    // Code
    logFile = fopen(file.c_str(), "w");
    if (logFile == nullptr)
    {
        MessageBox(NULL, TEXT("Failed To Create Log File ... Exiting !!!"), TEXT("File I/O Error"), MB_OK | MB_ICONERROR);
        exit(EXIT_FAILURE);
    }
}

Logger *Logger::getInstance(const std::string file)
{
    // Code
    if (_logger == nullptr)
        _logger = new Logger(file);

    return _logger;
}

void Logger::printLog(const char* fmt, ...)
{
    // Variable Declarations
    va_list argList;

    // Code
    if (logFile == nullptr)
        return;

    // Print Log Data
    va_start(argList, fmt);
    {
        vfprintf(logFile, fmt, argList);
    }
    va_end(argList);

    fflush(logFile);
}

void Logger::deleteInstance(void)
{
    delete _logger;
    _logger = nullptr;
}

Logger::~Logger()
{
    // Code
    if (logFile)
    {
        printLog("Log File Closed ...");
        
        fclose(logFile);
        logFile = nullptr;
    }
}
