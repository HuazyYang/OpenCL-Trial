#ifndef __CAMERA_HPP__
#define __CAMERA_HPP__
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class ArcBallCamera {
public:
  ArcBallCamera() {
    radius_ = 1.0f, pitch_ = .0f, yaw_ = .0f;
    UpdateViewMatrix();
  }
  ArcBallCamera(glm::vec3 center, float radius, float pitch, float yaw)
    : target_(center), radius_(radius), pitch_(pitch), yaw_(yaw_) {

    if(pitch_ > 89.0f) pitch_ = 89.0f;
    if(pitch_ < -89.0f) pitch_ = -89.0f;
    UpdateViewMatrix();
  }

  void SetPositions(glm::vec3 center, float radius, float pitch, float yaw) {
    target_ = center;
    radius_ = radius;
    pitch_ = pitch;
    yaw_ = yaw;
    if(pitch_ > 89.0f) pitch_ = 89.0f;
    if(pitch_ < -89.0f) pitch_ = -89.0f;
    UpdateViewMatrix();
  }

  float GetRadius() const { return radius_; }

  void Zoom(float dr, float rmin, float rmax) {
      const float scrollSpeed = 1.0f;

      radius_ += dr * scrollSpeed;

      if(radius_ < rmin) radius_ = rmin;
      if(radius_ > rmax) radius_ = rmax;
      UpdateViewMatrix();
  }

  void Rotate(float dx, float dy) {
    const float moveSpeed = 0.1f;
    yaw_ += moveSpeed * dx;
    pitch_ += moveSpeed * dy;

    if(pitch_ > 89.0f) pitch_ = 89.0f;
    if(pitch_ < -89.0f) pitch_ = -89.0f;

    UpdateViewMatrix();
  }

  const glm::mat4& GetView() const { return view_; }

  const glm::vec3& GetEyePosW() const { return eye_; }

private:
  ArcBallCamera(const ArcBallCamera &) = delete;
  void operator = (const ArcBallCamera &) = delete;

  void UpdateViewMatrix() {

    float theta = glm::radians(yaw_);
    float phi  = glm::radians(pitch_);

    eye_ = target_ - glm::vec3(
      radius_ * cos(theta) * cos(phi),
      radius_ * sin(phi),
      radius_ * sin(theta) * cos(phi)
    );

    view_ = glm::lookAt(eye_, target_, glm::vec3(.0f, 1.0f, .0f)) *
      glm::translate(glm::mat4(1.0f), -target_);
  }

  float radius_;
  float pitch_; // rotate about x axis
  float yaw_; // rotate about y axis
  glm::vec3 eye_;
  glm::vec3 target_;
  glm::mat4 view_;
};

#endif // __CAMERA_HPP__
