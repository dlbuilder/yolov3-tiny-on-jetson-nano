#pragma once
#include <vector>
#include <map>
using namespace std;

struct alignas(float) Detection {
	float bbox[4]; 	//x y w h
	float det_confidence;
	float class_id;
	float class_confidence;
};

enum class FilterStatus
{
	INIT,
	RUNNING,
	COMPLETE,
	ERR
};