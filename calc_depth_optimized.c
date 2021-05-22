/*
 * Project 2: Performance Optimization
 */

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
#endif

#if !defined(_MSC_VER)
#include <pthread.h>
#endif
#include <omp.h>

#include <math.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include "calc_depth_naive.h"
#include "calc_depth_optimized.h"
#include "utils.h"

float displacement_naive2(int dx, int dy) {
    float temp = dx * dx + dy * dy;
    float result = sqrt(temp);
    return result;
}

/* Helper function to return the square euclidean distance between two values. */
float square_euclidean_distance2(float a, float b) {
    int diff = a - b;
    return diff * diff;
}

void calc_depth_optimized(float *depth, float *left, float *right,
        int image_width, int image_height, int feature_width,
        int feature_height, int maximum_displacement) {
    
    // Naive implementation
    int max_iter = (2 * feature_width + 1)/4 * 4;
    int largest_val = 2 * feature_width + 1;

    memset(depth, 0, image_width * image_height * sizeof(int));

    #pragma omp parallel
    {	
    	#pragma omp for

	    for (int y = 0; y < image_height; y++) {
	        for (int x = 0; x < image_width; x++) {
	            if (y < feature_height || y >= image_height - feature_height
	                    || x < feature_width || x >= image_width - feature_width) {
	                depth[y * image_width + x] = 0;
	                continue;
	            }
	            float min_diff = -1;
	            int min_dy = 0;
	            int min_dx = 0;

	            for (int dy = -maximum_displacement; dy <= maximum_displacement; dy++) {
	                for (int dx = -maximum_displacement; dx <= maximum_displacement; dx++) {
	                    if (y + dy - feature_height < 0
	                            || y + dy + feature_height >= image_height
	                            || x + dx - feature_width < 0
	                            || x + dx + feature_width >= image_width) {
	                        continue;
	                    }
	                    float squared_diff = 0;
	                    float arr[4];
	                    __m128 sum = _mm_setzero_ps();

	                    //Unroll the for loop here and perform SIMD operations by 4 shifts
	                    for (int box_x = 0; box_x < max_iter; box_x+=4) {
	                        for (int box_y = -feature_height; box_y <= feature_height; box_y++) {
	                            
	                            int leftShift = (y + box_y) * image_width + (x + box_x - feature_width); 
	                            int rightShift = (y + dy + box_y) * image_width + (x + dx + box_x - feature_width); 

	                            //Performs the math to determine sum of squared differences for box
	                            __m128 temp_left = _mm_loadu_ps(left + leftShift); 
	                            __m128 temp_right = _mm_loadu_ps(right + rightShift); 
	                            __m128 sub = _mm_sub_ps(temp_left, temp_right);
	                            __m128 mul = _mm_mul_ps(sub, sub);

	                            sum = _mm_add_ps(mul, sum);
	                        }
	                    }

	                    //Take care of the tail case and perform operations by 1 shifts
	                    for (int box_x = max_iter; box_x < largest_val; box_x++) {
	                        for (int box_y = -feature_height; box_y <= feature_height; box_y++) {
	                            int left_x = x + box_x - feature_width;
	                            int left_y = y + box_y;
	                            int right_x = x + dx + box_x - feature_width;
	                            int right_y = y + dy + box_y;
	                            
	                            squared_diff += square_euclidean_distance2(
	                                    left[left_y * image_width + left_x],
	                                    right[right_y * image_width + right_x]
	                                    );
	                        }
	                    }
	                    _mm_storeu_ps(arr, sum);
	                    squared_diff += arr[0] + arr[1] + arr[2] + arr[3];

	                    if (min_diff == -1 || min_diff > squared_diff
	                            || (min_diff == squared_diff
	                                && displacement_naive2(dx, dy) < displacement_naive2(min_dx, min_dy))) {
	                        min_diff = squared_diff;
	                        min_dx = dx;
	                        min_dy = dy;
	                    }
	                }
	            }
	            if (min_diff != -1) {
	                if (maximum_displacement == 0) {
	                    depth[y * image_width + x] = 0;
	                } else {
	                    depth[y * image_width + x] = displacement_naive2(min_dx, min_dy);
	                }
	            } else {
	                depth[y * image_width + x] = 0;
	            }
	        }
	    }
	}	    
}