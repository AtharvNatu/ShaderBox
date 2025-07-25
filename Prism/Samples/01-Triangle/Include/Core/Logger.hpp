#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include <ctime>
#include <chrono>
#include <cstdarg>

class Logger
{
    private:
        FILE *logFile = nullptr;
        std::string getCurrentTime(void);

    protected:
        Logger(void);
        static Logger* _logger;

    public:
        //* Non-cloneable
        Logger(Logger &obj) = delete;

        //* Non-assignable
        void operator = (const Logger &) = delete;

        Logger(const std::string file);
        ~Logger();

        // Member Function Declarations
        static Logger* getInstance(const std::string file);
        void printLog(const char* fmt, ...);
        void deleteInstance(void);
};

#endif