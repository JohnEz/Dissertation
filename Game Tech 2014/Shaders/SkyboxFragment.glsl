#version 150 core

uniform samplerCube cubeTex;
uniform vec3 cameraPos;

in Vertex {
	vec3 normal;
} IN;

out vec4 gl_FragColor[2];

void main ( void ) {
	gl_FragColor[0] = texture(cubeTex, normalize(IN.normal));
	gl_FragColor[1] = vec4(1, 0, 0, 0);
}