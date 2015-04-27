#version 330 core

uniform vec4 color;
uniform sampler2D tex;

in vec2 UV;

out vec4 gl_FragColor;

void main(){
	gl_FragColor = color * texture2D(tex, UV);
}