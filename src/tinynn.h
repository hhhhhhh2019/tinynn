#ifndef NN_H
#define NN_H


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
	float* biases;
	float* biases_gradients;
	float* errors;
} Layer;


typedef struct {
	unsigned int count;
	Layer* layers;
	float*** weights;
	float*** gradients;

	float (*activation)(float);
	float (*derivative)(float);
} NN;


void nn_init(NN*, unsigned int, unsigned int[]);
void forward(NN*);

void clear_gradients(NN*);
void correct_weights(NN*, int);
void correct_biases(NN*, int);

void bp_backward(NN*);
void bp_count_gradients(NN*, float);

void rf_reward(NN*, float);


#endif // NN_H
