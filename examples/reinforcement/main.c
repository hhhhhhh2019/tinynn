#include <tinynn.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>


NN nn;


void test(float a, float b) {
	nn.layers[0].neurons[0] = a;
	nn.layers[0].neurons[1] = b;
	
	forward(&nn);

	printf("%f\n", nn.layers[nn.count-1].neurons[0]);
}


float numbers[10] = {
	0,1,
	0.5,0.75,
	0.1,0.3,
	0.5,1,
	0.6,0.6
};



int main() {
	srand(0);//time(NULL));
	nn_init(&nn, 3, (unsigned int[]){2,3,1});
	
	for (int i = 0; i < 100000; i++) {
		clear_gradients(&nn);

		for (int j = 0; j < 5; j++) {
			nn.layers[0].neurons[0] = numbers[j*2+0];
			nn.layers[0].neurons[1] = numbers[j*2+1];

			forward(&nn);

			rf_reward(&nn, 
					(numbers[j*2+0] + numbers[j*2+1]) / 2 -
					nn.layers[nn.count-1].neurons[0]);
		}

		correct_weights(&nn, 1);
	}

	for (int i = 0; i < 5; i++)
		test(numbers[i*2+0], numbers[i*2+1]);
}
