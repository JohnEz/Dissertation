#version 150 core

uniform mat4 PVMmat;        // The projection-view-model matrix
uniform vec4 charCoords;    // The CharCoord struct for the character you are rendering, {x, y, w, h}
uniform float texSize;      // The size of the texture which contains the rasterized characters (assuming it is square)
uniform vec2 offset;        // The offset at which to paint, w.r.t the first character

attribute vec2 vertex;

// Output data ; will be interpolated for each fragment.
out vec2 UV;

void main(){

	UV = (charCoords.xy + charCoords.zw * vec2(vertex.x, 1. - vertex.y)) / texSize;
	float x = (charCoords[2] * vertex.x + offset.x) / charCoords[3];
	float y = vertex.y + offset.y / charCoords[3];
	gl_Position = PVMmat * vec4(x, y, 0., 1.);
}

