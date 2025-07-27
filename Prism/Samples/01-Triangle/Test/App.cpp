#include "Prism.hpp"

class App : public Prism
{
    public:

        void initialize()
        {

        }

        void display()
        {

        }

        void update()
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
