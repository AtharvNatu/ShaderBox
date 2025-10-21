#include "Camera.hpp"

Camera::Camera(vmath::vec3 _position, vmath::vec3 _up)
{
    position = _position;
    worldUp = _up;
    front = vmath::vec3(0.0f, 0.0f, -1.0f);
    yaw = YAW;
    pitch = PITCH;
    movementSpeed = SPEED;
    mouseSensitivity = SENSITIVITY;
    zoom = ZOOM;

    updateCameraVectors();
}

Camera::Camera(
    float posX, float posY, float posZ, 
    float upX, float upY, float upZ, 
    float _yaw, float _pitch
)
{
    position = vmath::vec3(posX, posY, posZ);
    worldUp = vmath::vec3(upX, upY, upZ);
    front = vmath::vec3(0.0f, 0.0f, -1.0f);
    yaw = _yaw;
    pitch = _pitch;
    movementSpeed = SPEED;
    mouseSensitivity = SENSITIVITY;
    zoom = ZOOM;

    updateCameraVectors();
}

vmath::mat4 Camera::getViewMatrix() const
{
    return vmath::lookat(position, position + front, up);
}

void Camera::processKeyboard(CAMERA_DIRECTION direction, float deltaTime)
{
    float velocity = movementSpeed * deltaTime;

    switch(direction)
    {
        case FORWARD:
            position += front * velocity;
        break;

        case BACKWARD:
            position -= front * velocity;
        break;

        case LEFT:
            position -= right * velocity;
        break;

        case RIGHT:
            position += right * velocity;
        break;

        case UP:
            position += worldUp * velocity;
        break;

        case DOWN:
            position -= worldUp * velocity;
        break;
    }
}

void Camera::processMouseMovement(float xOffset, float yOffset, bool constrainPitch)
{
    xOffset *= mouseSensitivity;
    yOffset *= mouseSensitivity;

    yaw += xOffset;
    pitch += yOffset;

    if (constrainPitch)
    {
        if (pitch > 89.0f)
            pitch = 89.0f;
        if (pitch < -89.0f)
            pitch = -89.0f;
    }

    updateCameraVectors();
}

void Camera::processMouseScroll(float yOffset)
{
    zoom -= (float)yOffset;
    if (zoom < 1.0f)
        zoom = 1.0f;
    if (zoom > 45.0f)
        zoom = 45.0f;
}

void Camera::updateCameraVectors()
{
    front[0] = cos(vmath::radians(yaw)) * cos(vmath::radians(pitch));
    front[1] = sin(vmath::radians(pitch));
    front[2] = sin(vmath::radians(yaw)) * cos(vmath::radians(pitch));

    front = vmath::normalize(front);
    right = vmath::normalize(vmath::cross(front, worldUp));
    up = vmath::normalize(vmath::cross(right, front));
}

