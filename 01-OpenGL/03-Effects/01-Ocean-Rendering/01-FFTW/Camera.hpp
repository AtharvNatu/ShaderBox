#ifndef CAMERA_HPP
#define CAMERA_HPP

//! OpenGL Header Files
#include <GL/glew.h>
#include <GL/gl.h>

#include "vmath.h"

enum CAMERA_DIRECTION
{
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT
};

const float YAW = -90.0f;
const float PITCH = 0.0f;
const float SPEED = 2.5f;
const float SENSITIVITY = 0.1f;
const float ZOOM = 45.0f;

class Camera
{
    private:

        //* Camera Attributes
        vmath::vec3 position;
        vmath::vec3 front;
        vmath::vec3 up;
        vmath::vec3 right;
        vmath::vec3 worldUp;

        //* Euler Angles
        float yaw;
        float pitch;

        void updateCameraVectors();

    public:
        //* Camera Options
        float movementSpeed;
        float mouseSensitivity;
        float zoom;

        Camera(vmath::vec3 position, vmath::vec3 up);
        Camera(
            float posX, float posY, float posZ, 
            float upX, float upY, float upZ, 
            float yaw, float pitch
        );

        vmath::mat4 getViewMatrix();

        void processKeyboard(CAMERA_DIRECTION direction, float deltaTime);
        void processMouseMovement(float xOffset, float yOffset, bool constrainPitch);
        void processMouseScroll(float yOffset);
};


#endif // CAMERA_HPP
