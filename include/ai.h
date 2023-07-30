#ifndef AI_H
#define AI_H


// ---------- WEIGHTS ----------
//
// Wxy
//   x - индекс нейрона с предыдущего слоя
//   y - индекс нейрона со следующего слоя
//
//    w11
// n1 --- n1
//      /
//     / w21
// n2 /
//
//  x  1   2
// y
// 1 w11 w21
//
// -----------------------------


typedef struct {
	unsigned int count;
	float* neurons;
	float* errors;
} Layer;


typedef struct {
	unsigned int count;
	Layer* layers;
	float*** weights;
	float*** gradients;

	float (*activation)(float);
	float (*derivative)(float);
} AI;


void ai_init(AI*, unsigned int, unsigned int[]);
void forward(AI*);
void backward(AI*);
void clear_gradients(AI*);
void count_gradients(AI*, float);
void correct_weights(AI*, int);


#endif // AI_H
