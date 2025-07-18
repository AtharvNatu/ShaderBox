#ifndef MODEL_VIEW_MATRIX_HPP
#define MODEL_VIEW_MATRIX_HPP

#include "vmath.h"
#include <iostream>

extern FILE* gpFile;

class ModelViewMatrixStack
{
    private:
        static const int STACK_SIZE = 32;
        vmath::mat4 matrixStack[STACK_SIZE];
        int matrixStackTop = -1;
    
    public:
        ModelViewMatrixStack();
        void pushMatrix(vmath::mat4 matrix);
        vmath::mat4 popMatrix();

};

#endif // MODEL_VIEW_MATRIX_HPP