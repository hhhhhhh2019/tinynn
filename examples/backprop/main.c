#include <tinynn.h>
#include <stdio.h>


NN nn;


int main() {
	nn_init(&nn, 3, (unsigned int[]){2,2,1});

	for (int i = 0; i < 10000; i++) {
		float err = 0;

		clear_gradients(&nn);
		
		for (int j = 0; j < 4; j++) {
			nn.layers[0].neurons[0] = j%2;
			nn.layers[0].neurons[1] = j/2;

			forward(&nn);

			nn.layers[nn.count-1].errors[0] = (float)((j%2)^(j/2)) - nn.layers[nn.count-1].neurons[0];

			err += nn.layers[nn.count-1].errors[0]*nn.layers[nn.count-1].errors[0];

			bp_backward(&nn);
			bp_count_gradients(&nn, 0.5);
		}

		correct_weights(&nn, 4);
		correct_biases(&nn, 4);

		printf("%d %f\n", i, err/4);
	}


	for (int i = 0; i < 4; i++) {
		nn.layers[0].neurons[0] = i%2;
		nn.layers[0].neurons[1] = i/2;

		forward(&nn);

		fprintf(stderr, "%f\n", nn.layers[nn.count-1].neurons[0]);
	}
}
