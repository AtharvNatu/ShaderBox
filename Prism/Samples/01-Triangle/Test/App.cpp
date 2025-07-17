#include "Prism.hpp"

class App : public Prism
{
    public:
        void display()
        {

        }

        void update()
        {

        }

        void initialize()
        {

        }

        void uninitialize()
        {
            
        }
};

int main(int argc, char** argv)
{
    Prism prism(API::OpenGL, "Prism OpenGL Sample", 800, 600);

    return prism.exec();
}
