#ifndef CAMERA_HPP
#define CAMERA_HPP

//! GLM Related Macros and Header Files
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <glm/gtc/matrix_transform.hpp>

enum CAMERA_DIRECTION
{
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    UP,
    DOWN
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
        glm::vec3 position;
        glm::vec3 front;
        glm::vec3 up;
        glm::vec3 right;
        glm::vec3 worldUp;

        //* Euler Angles
        float yaw;
        float pitch;

        void updateCameraVectors();

    public:
        //* Camera Options
        float movementSpeed;
        float mouseSensitivity;
        float zoom;

        Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f));
        Camera(
            float posX, float posY, float posZ, 
            float upX, float upY, float upZ, 
            float yaw, float pitch
        );

        glm::mat4 getViewMatrix() const;

        void processKeyboard(CAMERA_DIRECTION direction, float deltaTime);
        void processMouseMovement(float xOffset, float yOffset, bool constrainPitch = true);
        void processMouseScroll(float yOffset);
};


#endif // CAMERA_HPP
