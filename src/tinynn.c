#include "tinynn.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>


float _def_activation(float x) {
	return 1/(1+exp(-x));
}

float _def_derivative(float y) {
	return y*(1.0-y);
}


void nn_init(NN* nn, unsigned int count, unsigned int neurons[]) {
	nn->activation = &_def_activation;
	nn->derivative = &_def_derivative;

	nn->count = count;
	nn->layers = calloc(count * sizeof(Layer), 1);

	for (int i = 0; i < count; i++) {
		nn->layers[i].count = neurons[i];
		nn->layers[i].neurons          = calloc(neurons[i] * sizeof(float), 1);
		nn->layers[i].biases           = calloc(neurons[i] * sizeof(float), 1);
		nn->layers[i].biases_gradients = calloc(neurons[i] * sizeof(float), 1);
		nn->layers[i].errors           = calloc(neurons[i] * sizeof(float), 1);

		for (int j = 0; j < neurons[i]; j++) {
			nn->layers[i].biases[j] = (float)rand() / RAND_MAX * 2 - 1;
		}
	}

	nn->weights   = malloc(sizeof(float**)*(count-1));
	nn->gradients = malloc(sizeof(float**)*(count-1));

	for (int i = 0; i < count-1; i++) {
		nn->weights[i]   = malloc(sizeof(float*)*neurons[i]);
		nn->gradients[i] = malloc(sizeof(float*)*neurons[i]);

		for (int j = 0; j < neurons[i]; j++) {
			nn->weights[i][j]   = malloc(sizeof(float)*neurons[i+1]);
			nn->gradients[i][j] = malloc(sizeof(float)*neurons[i+1]);

			for (int k = 0; k < neurons[i+1]; k++)
				nn->weights[i][j][k] = (float)rand() / RAND_MAX * 2 - 1;
		}
	}
}


void forward(NN* nn) {
	for (int i = 1; i < nn->count; i++) {
		for (int j = 0; j < nn->layers[i].count; j++) {
			float val = 0;

			for (int k = 0; k < nn->layers[i-1].count; k++)
				val += nn->layers[i-1].neurons[k] * nn->weights[i-1][k][j];

			nn->layers[i].neurons[j] = (*nn->activation)(val+nn->layers[i].biases[j]);
		}
	}
}


void clear_gradients(NN* nn) {
	for (int i = 0; i < nn->count-1; i++) {
		for (int j = 0; j < nn->layers[i].count; j++) {
			for (int k = 0; k < nn->layers[i+1].count; k++) {
				nn->gradients[i][j][k] = 0;
			}
		}
	}

	for (int i = 0; i < nn->count; i++) {
		for (int j = 0; j < nn->layers[i].count; j++) {
			nn->layers[i].biases_gradients[j] = 0;
		}
	}
}


void correct_weights(NN* nn, int count) {
	for (int i = 0; i < nn->count-1; i++) {
		for (int j = 0; j < nn->layers[i].count; j++) {
			for (int k = 0; k < nn->layers[i+1].count; k++) {
				nn->weights[i][j][k] += nn->gradients[i][j][k] / (float)count;
			}
		}
	}
}


void correct_biases(NN* nn, int count) {
	for (int i = 0; i < nn->count; i++) {
		for (int j = 0; j < nn->layers[i].count; j++) {
			nn->layers[i].biases[j] += nn->layers[i].biases_gradients[j] / (float)count;
		}
	}
}


void bp_backward(NN* nn) {
	for (int i = nn->count-2; i >= 0; i--) {
		for (int j = 0; j < nn->layers[i].count; j++) {
			float err = 0;

			for (int k = 0; k < nn->layers[i+1].count; k++)
				err += nn->layers[i+1].errors[k] * nn->weights[i][j][k];

			nn->layers[i].errors[j] = err;
		}
	}
}


void bp_count_gradients(NN* nn, float koof) {
	for (int i = 0; i < nn->count-1; i++) {
		for (int j = 0; j < nn->layers[i].count; j++) {
			for (int k = 0; k < nn->layers[i+1].count; k++) {
				nn->gradients[i][j][k] += koof * nn->layers[i+1].errors[k]*2 * (*nn->derivative)(nn->layers[i+1].neurons[k]) * nn->layers[i].neurons[j];
			}
		}
	}

	for (int i = 0; i < nn->count; i++) {
		for (int j = 0; j < nn->layers[i].count; j++) {
			nn->layers[i].biases_gradients[j] += koof * nn->layers[i].errors[j] * (*nn->derivative)(nn->layers[i].neurons[j]);
		}
	}
}


void rf_reward(NN* nn, float reward) {
	for (int i = 0; i < nn->count-1; i++) {
		for (int j = 0; j < nn->layers[i].count; j++) {
			for (int k = 0; k < nn->layers[i+1].count; k++) {
				nn->gradients[i][j][k] += reward;// * nn->weights[i][j][k] * nn->layers[i].neurons[j] * (*nn->derivative)(nn->layers[i+1].neurons[k]);
			}
		}
	}
}
