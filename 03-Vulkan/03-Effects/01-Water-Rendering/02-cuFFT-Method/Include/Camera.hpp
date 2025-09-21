#ifndef CAMERA_HPP
#define CAMERA_HPP

//! GLM Related Macros and Header Files
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

enum Direction
{
    FORWARD = 1,
    BACKWARD = 2,
    LEFT = 4,
    RIGHT = 8
};

class Camera
{
    private:
        glm::vec3 position;
        float yaw, pitch;
        float fovY, aspectRatio, nearClip, farClip;
        float rotationSpeed;
        float movementSpeed;

    public:
        Camera(
            glm::vec3 cPosition,
            float rYaw,
            float rPitch,
            float dFovY,
            float fAspectRatio,
            float fNearClip,
            float fFarClip,
            float fRotationSpeed,
            float fMoveSpeed
        ) : position(cPosition),
            yaw(rYaw),
            pitch(rPitch),
            fovY(dFovY),
            aspectRatio(fAspectRatio),
            nearClip(fNearClip),
            farClip(fFarClip),
            rotationSpeed(fRotationSpeed),
            movementSpeed(fMoveSpeed){ };

        inline glm::vec3 getPosition() const
        {
            return this->position;
        }

        inline void setAspectRatio(float ratio)
        {
            this->aspectRatio = ratio;
        }

        inline void rotateYaw(float dt)
        {
            this->yaw += dt * rotationSpeed;
        }

        inline void roatatePitch(float dt)
        {
            this->pitch += dt * rotationSpeed;
        }

        inline void setMovementSpeed(float speed)
        {
            this->movementSpeed = speed;
        }
    
        inline void setRotationSpeed(float speed)
        {
            this->rotationSpeed = speed;
        }

        glm::mat4 getViewProjection(bool translate) const;
        glm::mat4 getRotationMatrix() const;
        void move(float dt, int direction);
};


#endif // CAMERA_HPP