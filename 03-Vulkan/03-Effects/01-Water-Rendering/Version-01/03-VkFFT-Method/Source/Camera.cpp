#include "Camera.hpp"

glm::mat4 Camera::getViewProjection(bool translate) const
{
    glm::mat4 rotationMatrix = getRotationMatrix();
    glm::mat4 translationMatrix = glm::mat4(1.0f);

    if (translate)
    {
        translationMatrix = glm::translate(translationMatrix, position);
    }

    glm::mat4 projectionMatrix = glm::perspective(
        this->fovY,
        this->aspectRatio,
        this->nearClip,
        this->farClip
    );

    return projectionMatrix * glm::inverse(translationMatrix * rotationMatrix);
}


glm::mat4 Camera::getRotationMatrix() const 
{
    glm::mat4 rotate = glm::rotate(
        glm::mat4(1.0), 
        glm::radians<float>(this->yaw), 
        glm::vec3(0.0, 1.0, 0.0)
    );

    return glm::rotate(
        rotate, 
        glm::radians<float>(this->pitch), 
        glm::vec3(1.0, 0.0, 0.0)
    );
}

void Camera::move(float dt, int direction) 
{
    glm::mat4 rotation = getRotationMatrix();
    glm::vec3 forward = glm::vec3(rotation * glm::vec4(0.0, 0.0, -1.0, 0.0));
    glm::vec3 right = glm::vec3(rotation * glm::vec4(1.0, 0.0, 0.0, 0.0));
    
    if (direction & Direction::FORWARD)
        this->position += forward * (float) (dt * this->movementSpeed);
    if (direction & Direction::BACKWARD)
        this->position -= forward * (float) (dt * this->movementSpeed);
    if (direction & Direction::RIGHT)
        this->position += right * (float) (dt * this->movementSpeed);
    if (direction & Direction::LEFT)
        this->position -= right * (float) (dt * this->movementSpeed);
}

