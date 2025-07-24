#ifndef LOAD_MESH_H
#define LOAD_MESH_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <GL/glew.h>
#include <GL/gl.h>

#define SUCCESS         1
#define FAILURE         -1
#define BUFFER_SIZE     1024

// Variable Declarations
typedef struct TAG_vecInt
{
    GLint* ptr = NULL;
    size_t size;
} vecInt;

typedef struct TAG_vecFloat
{
    GLfloat* ptr = NULL;
    size_t size;
} vecFloat;

GLchar buffer[BUFFER_SIZE];

FILE* gpMeshFile = NULL;

vecFloat *gpVertex, *gpTexture, *gpNormal;
vecInt *gpVertexIndices, *gpTextureIndices, *gpNormalIndices;

// Code
vecInt* createIntVector(void)
{
    vecInt* ptr = (vecInt*)malloc(sizeof(vecInt));
    if (ptr == NULL)
        return NULL;

    memset(ptr, 0, sizeof(vecInt));
    return ptr;
}

vecFloat* createFloatVector(void)
{
    vecFloat* ptr = (vecFloat*)malloc(sizeof(vecFloat));
    if (ptr == NULL)
        return NULL;

    memset(ptr, 0, sizeof(vecFloat));
    return ptr;
}

int pushBackToIntVector(vecInt* ptrVecInt, int data)
{
    ptrVecInt->ptr = (GLint*)realloc(ptrVecInt->ptr, (ptrVecInt->size + 1) * sizeof(GLint));

    if (ptrVecInt->ptr == NULL)
        return FAILURE;
    
    ptrVecInt->size = ptrVecInt->size + 1;
    ptrVecInt->ptr[ptrVecInt->size - 1] = data;

    return SUCCESS;
}

int pushBackToFloatVector(vecFloat* ptrVecFloat, float data)
{
    ptrVecFloat->ptr = (GLfloat*)realloc(ptrVecFloat->ptr, (ptrVecFloat->size + 1) * sizeof(GLfloat));

    if (ptrVecFloat->ptr == NULL)
        return FAILURE;
    
    ptrVecFloat->size = ptrVecFloat->size + 1;
    ptrVecFloat->ptr[ptrVecFloat->size - 1] = data;

    return SUCCESS;
}

int destroyIntVector(vecInt* ptrVecInt)
{
    if (ptrVecInt->ptr)
    {
        free(ptrVecInt->ptr);
        free(ptrVecInt);
        return SUCCESS;
    }

    return FAILURE;
}

int destroyFloatVector(vecFloat* ptrVecFloat)
{
    if (ptrVecFloat->ptr)
    {
        free(ptrVecFloat->ptr);
        free(ptrVecFloat);
        return SUCCESS;
    }

    return FAILURE;
}

int loadMesh(const char* file)
{
    // Variable Declarations
    GLchar* space = " ";
    GLchar* slash = "/";
    GLchar* firstToken = NULL;
    GLchar* token;

    GLchar* faceEntries[3] = { NULL, NULL, NULL };
    GLint nrPosCords = 0, nrTexCords = 0, nrNormalCords = 0, nrFaces = 0;

    // Code
    gpMeshFile = fopen(file, "r");
    if (gpMeshFile == NULL)
        return FAILURE;

    gpVertex = createFloatVector();
    gpTexture = createFloatVector();
    gpNormal = createFloatVector();

    gpVertexIndices = createIntVector();
    gpTextureIndices = createIntVector();
    gpNormalIndices = createIntVector();

    while (fgets(buffer, BUFFER_SIZE, gpMeshFile) != NULL)
	{
		firstToken = strtok(buffer, space);

		if (strcmp(firstToken, "v") == 0)
		{
			nrPosCords++;
			while ((token = strtok(NULL, space)) != NULL)
				pushBackToFloatVector(gpVertex, atof(token));
		}

		else if (strcmp(firstToken, "vt") == 0)
		{
			nrTexCords++;
			while ((token = strtok(NULL, space)) != NULL)
				pushBackToFloatVector(gpTexture, atof(token));
		}

		else if (strcmp(firstToken, "vn") == 0)
		{
			nrNormalCords++;
			while ((token = strtok(NULL, space)) != NULL)
				pushBackToFloatVector(gpNormal, atof(token));
		}

		else if (strcmp(firstToken, "f") == 0)
		{
			nrFaces++;
			for (int i = 0; i < 3; i++)
				faceEntries[i] = strtok(NULL, space);

			for (int i = 0; i < 3; i++)
			{
				token = strtok(faceEntries[i], slash);
				pushBackToIntVector(gpVertexIndices, atoi(token) - 1);

				token = strtok(NULL, slash);
				pushBackToIntVector(gpTextureIndices, atoi(token) - 1);

				token = strtok(NULL, slash);
				pushBackToIntVector(gpNormalIndices, atoi(token) - 1);
			}
		}
	}

    fclose(gpMeshFile);
    gpMeshFile = NULL;

    return SUCCESS;
}


#endif  // LOAD_MESH_H
