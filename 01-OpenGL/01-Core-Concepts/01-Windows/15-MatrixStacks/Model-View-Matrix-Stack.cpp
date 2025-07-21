#include "Model-View-Matrix-Stack.hpp"

ModelViewMatrixStack::ModelViewMatrixStack()
{
    // Code
    matrixStackTop = 0;

    for (int i = 0; i < STACK_SIZE; i++)
        matrixStack[i] = vmath::mat4::identity();
}

void ModelViewMatrixStack::pushMatrix(vmath::mat4 matrix)
{
    // Code
    fprintf(gpFile, "Before PUSH, Stack Top = %d\n", matrixStackTop);

    if (matrixStackTop > (STACK_SIZE - 1))
        fprintf(gpFile, "Stack Overflow Occurred : Exceeded Matrix Stack Limit !!!\n");
    
    matrixStack[matrixStackTop] = matrix;
    matrixStackTop++;

    fprintf(gpFile, "Before PUSH, Stack Top = %d\n", matrixStackTop);
}

vmath::mat4 ModelViewMatrixStack::popMatrix()
{
    // Variable Declarations
    vmath::mat4 matrix;

    // Code
    fprintf(gpFile, "Before POP, Stack Top = %d\n", matrixStackTop);

	if (matrixStackTop < 0)
		fprintf(gpFile, "Stack Underflow Occurred : Matrix Stack Empty !!!\n");

	matrixStack[matrixStackTop] = vmath::mat4::identity();
	matrixStackTop--;

	matrix = matrixStack[matrixStackTop];

	fprintf(gpFile, "After POP, Stack Top = %d\n", matrixStackTop);

	return matrix;
}

