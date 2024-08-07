#pragma once
#include <vector>

void normal_(std::vector<float>& nums, float mean, float std);
void uniform_(std::vector<float>& nums, float min, float max);
std::vector<float> normal(int64_t size, float mean, float std);
std::vector<float> uniform(int64_t size, float min, float max);
