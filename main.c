#include <ai.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


AI nn;


int main() {
	srand(time(NULL));
	ai_init(&nn, 3, (unsigned int[]){2,5,1});

	/*for (int i = 0; i < nn.count-1; i++) {
		for (int j = 0; j < nn.layers[i].count; j++) {
			for (int k = 0; k < nn.layers[i+1].count; k++)
				fprintf(stderr, "%f ", nn.weights[i][j][k]);
			putc('\n', stderr);
		}

		putc('\n', stderr);
	}*/

	for (int i = 0; i < 100000; i++) {
		float e = 0;

		for (int j = 0; j < 4; j++) {
			nn.layers[0].neurons[0] = j%2;
			nn.layers[0].neurons[1] = j/2;

			forward(&nn);

			nn.layers[nn.count-1].errors[0] = (float)((j%2)^(j/2)) - nn.layers[nn.count-1].neurons[0];
			//printf("%d %f\n", i, nn.layers[nn.count-1].errors[0]*nn.layers[nn.count-1].errors[0]);
			e += nn.layers[nn.count-1].errors[0]*nn.layers[nn.count-1].errors[0];

			backward(&nn);
			correct_weights(&nn, 0.5);
		}

		printf("%d %f\n", i, e/4);
	}


	for (int i = 0; i < 4; i++) {
		nn.layers[0].neurons[0] = i%2;
		nn.layers[0].neurons[1] = i/2;

		forward(&nn);

		for (int k = 0; k < nn.layers[nn.count-1].count; k++)
			fprintf(stderr, "%f ", nn.layers[nn.count-1].neurons[k]);

		putc('\n', stderr);
	}
}
