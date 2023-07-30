#include <ai.h>
#include <stdlib.h>
#include <math.h>


float _def_activation(float x) {
	return 1/(1+exp(-x));
}

float _def_derivative(float y) {
	return y*(1.0-y);
}


void ai_init(AI* ai, unsigned int count, unsigned int neurons[]) {
	ai->activation = &_def_activation;
	ai->derivative = &_def_derivative;

	ai->count = count;
	ai->layers = calloc(count * sizeof(Layer), 1);

	for (int i = 0; i < count; i++) {
		ai->layers[i].count = neurons[i];
		ai->layers[i].neurons = calloc(neurons[i] * sizeof(float), 1);
		ai->layers[i].errors  = calloc(neurons[i] * sizeof(float), 1);
	}

	ai->weights   = malloc(sizeof(float**)*(count-1));
	ai->gradients = malloc(sizeof(float**)*(count-1));

	for (int i = 0; i < count-1; i++) {
		ai->weights[i]   = malloc(sizeof(float*)*neurons[i]);
		ai->gradients[i] = malloc(sizeof(float*)*neurons[i]);

		for (int j = 0; j < neurons[i]; j++) {
			ai->weights[i][j]   = malloc(sizeof(float)*neurons[i+1]);
			ai->gradients[i][j] = malloc(sizeof(float)*neurons[i+1]);

			for (int k = 0; k < neurons[i+1]; k++)
				ai->weights[i][j][k] = (float)rand() / RAND_MAX * 2 - 1;
		}
	}
}


void forward(AI* ai) {
	for (int i = 1; i < ai->count; i++) {
		for (int j = 0; j < ai->layers[i].count; j++) {
			float val = 0;

			for (int k = 0; k < ai->layers[i-1].count; k++)
				val += ai->layers[i-1].neurons[k] * ai->weights[i-1][k][j];

			ai->layers[i].neurons[j] = (*ai->activation)(val);
		}
	}
}


void backward(AI* ai) {
	for (int i = ai->count-2; i >= 0; i--) {
		for (int j = 0; j < ai->layers[i].count; j++) {
			float err = 0;

			for (int k = 0; k < ai->layers[i+1].count; k++)
				err += ai->layers[i+1].errors[k] * ai->weights[i][j][k];

			ai->layers[i].errors[j] = err;
		}
	}
}


void clear_gradients(AI* ai) {
	for (int i = 0; i < ai->count-1; i++) {
		for (int j = 0; j < ai->layers[i].count; j++) {
			for (int k = 0; k < ai->layers[i+1].count; k++) {
				ai->gradients[i][j][k] = 0;
			}
		}
	}
}


void count_gradients(AI* ai, float koof) {
	for (int i = 0; i < ai->count-1; i++) {
		for (int j = 0; j < ai->layers[i].count; j++) {
			for (int k = 0; k < ai->layers[i+1].count; k++) {
				ai->gradients[i][j][k] += koof * ai->layers[i+1].errors[k] * (*ai->derivative)(ai->layers[i+1].neurons[k]) * ai->layers[i].neurons[j];
			}
		}
	}
}


void correct_weights(AI* ai) {
	for (int i = 0; i < ai->count-1; i++) {
		for (int j = 0; j < ai->layers[i].count; j++) {
			for (int k = 0; k < ai->layers[i+1].count; k++) {
				ai->weights[i][j][k] += ai->gradients[i][j][k];
				//koof * ai->layers[i+1].errors[k] * (*ai->derivative)(ai->layers[i+1].neurons[k]) * ai->layers[i].neurons[j];
			}
		}
	}
}
