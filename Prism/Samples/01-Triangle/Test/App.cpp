#include "Prism.hpp"

// class App : public Prism
// {
//     public:

//         INIT_STATUS initialize() override
//         {
//             Prism::initialize();
//         }

//         void display() override
//         {
//             Prism::display();
//         }

//         void update() override
//         {
//             Prism::update();
//         }

//         void uninitialize() override
//         {
//             Prism::uninitialize();
//         }
// };

int main(int argc, char** argv)
{
    // Prism prism(API::OpenGL, "Prism OpenGL Sample", 800, 600);
    Prism prism;
    prism.setAPI(API::OpenGL);
    prism.setWindowTitle("Prism OpenGL Sample");
    prism.setWindowWidth(800);
    prism.setWindowHeight(600);

    prism.initialize();

    return prism.exec();
}
