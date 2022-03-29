#pragma once
#include<assimp/Quaternion.h>
#include<assimp/vector3.h>
#include<assimp/matrix4x4.h>
#include<glm/glm.hpp>
#include<glm/gtc/quaternion.hpp>

namespace AssimpGLMHelpers {

inline
glm::mat4 aiMatrix4x4ToGLMmat4(const aiMatrix4x4 &from) {

  return glm::mat4{
    from.a1, from.a2, from.a3, from.a4,
    from.b1, from.b2, from.b3, from.b4,
    from.c1, from.c2, from.c3, from.c4,
    from.d1, from.d2, from.d3, from.d4
  };

  	// 	glm::mat4 to;
		// //the a,b,c,d in assimp is the row ; the 1,2,3,4 is the column
		// to[0][0] = from.a1; to[1][0] = from.a2; to[2][0] = from.a3; to[3][0] = from.a4;
		// to[0][1] = from.b1; to[1][1] = from.b2; to[2][1] = from.b3; to[3][1] = from.b4;
		// to[0][2] = from.c1; to[1][2] = from.c2; to[2][2] = from.c3; to[3][2] = from.c4;
		// to[0][3] = from.d1; to[1][3] = from.d2; to[2][3] = from.d3; to[3][3] = from.d4;
		// return to;
}

inline
glm::vec3 aiVector3DToGLMvec3(const aiVector3D &aiv) {
  return glm::vec3(aiv.x, aiv.y, aiv.z);
}

inline
glm::quat aiQuaternionToGLMquat(const aiQuaternion &aiq) {
  return glm::quat(aiq.w, aiq.x, aiq.y, aiq.z);
}

}