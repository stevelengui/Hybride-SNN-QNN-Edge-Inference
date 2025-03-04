#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include "weights.h"

#define DEBUG 1
#define TAU 0.85f
#define TEMPERATURE 1.0f
#define REFRACTORY_PERIOD 2
#define DROPOUT_RATE 0.5f
#define LABEL_SMOOTHING 0.1f
#define NOISE_SCALE 0.05f
#define SPARSITY_WEIGHT 0.005f

// Neuron structure
typedef struct {
    float *mem;
    float base_threshold;
    float scale;
    float zero_point;
    float adaptive_threshold;
    int refractory_period;
} HybridNeuron;

// LSTM state structure
typedef struct {
    float *hidden_state;
    float *cell_state;
} LSTMState;

// Function to get current time in milliseconds
long long get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

// Initialize neuron
void init_neuron(HybridNeuron *neuron, int size, float threshold, float scale, float zero_point) {
    neuron->mem = (float *)calloc(size, sizeof(float));
    if (neuron->mem == NULL) {
        fprintf(stderr, "Memory allocation failed for neuron->mem\n");
        exit(1);
    }
    neuron->base_threshold = threshold;
    neuron->scale = scale;
    neuron->zero_point = zero_point;
    neuron->refractory_period = 0;

    #if DEBUG
    printf("Neuron: scale=%.4f, zero=%.4f, base_thr=%.4f\n",
           neuron->scale, neuron->zero_point, neuron->base_threshold);
    #endif
}

// Hybrid neuron forward pass
float hybrid_neuron_forward(HybridNeuron *neuron, float input, int idx) {
    if (neuron->refractory_period > 0) {
        neuron->refractory_period--;
        return 0.0f;
    }

    // Adaptive quantization
    int bit_width = (neuron->scale > 0.5f) ? 8 : 4;
    float q_range = powf(2, bit_width) - 1;
    float quantized = roundf(input * (q_range / (2 * neuron->scale))) / (q_range / (2 * neuron->scale));

    // Membrane update with leakage
    neuron->mem[idx] = TAU * neuron->mem[idx] + quantized;

    // Dynamic threshold adaptation with decay
    neuron->adaptive_threshold = neuron->base_threshold + 0.05f * fabsf(neuron->mem[idx]);
    neuron->adaptive_threshold *= 0.95f;  // Decay factor for stability

    float spike = (neuron->mem[idx] > neuron->adaptive_threshold) ? 1.0f : 0.0f;

    // Soft reset mechanism
    if (spike > 0.5f) {
        neuron->mem[idx] = -neuron->base_threshold * 0.5f;
        neuron->refractory_period = REFRACTORY_PERIOD;
    }

    #if DEBUG
    if (idx == 0) printf("In=%.3f Q=%.3f Mem=%.3f Thr=%.3f Spike=%.0f\n",
                       input, quantized, neuron->mem[idx],
                       neuron->adaptive_threshold, spike);
    #endif
    return spike;
}

// LSTM forward pass with dropout
void lstm_forward(const float *input, float *output, LSTMState *state) {
    const int hidden = LSTM_HIDDEN_SIZE;
    for (int i = 0; i < hidden; i++) {
        float gates[4] = {0};

        // Combined gate computation
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            gates[0] += input[j] * lstm_weight_ih[i * HIDDEN_SIZE + j];
            gates[1] += input[j] * lstm_weight_ih[(i + hidden) * HIDDEN_SIZE + j];
            gates[2] += input[j] * lstm_weight_ih[(i + 2 * hidden) * HIDDEN_SIZE + j];
            gates[3] += input[j] * lstm_weight_ih[(i + 3 * hidden) * HIDDEN_SIZE + j];
        }

        // Apply dropout to gates
        for (int k = 0; k < 4; k++) {
            if ((float)rand() / RAND_MAX < DROPOUT_RATE) gates[k] = 0.0f;
        }

        // State clamping
        state->cell_state[i] = fmaxf(fminf(state->cell_state[i], 5.0f), -5.0f);
        state->hidden_state[i] = fmaxf(fminf(state->hidden_state[i], 3.0f), -3.0f);

        // Gate computations with gradient clipping
        float ig = 1.0f / (1.0f + expf(-fmaxf(fminf(gates[0], 5.0f), -5.0f)));
        float fg = 1.0f / (1.0f + expf(-fmaxf(fminf(gates[1], 5.0f), -5.0f)));
        float cg = tanhf(fmaxf(fminf(gates[2], 2.5f), -2.5f));
        float og = 1.0f / (1.0f + expf(-fmaxf(fminf(gates[3], 5.0f), -5.0f)));

        state->cell_state[i] = fg * state->cell_state[i] + ig * cg;
        output[i] = og * tanhf(state->cell_state[i]);
    }

    #if DEBUG
    printf("LSTM: out0=%.3f cell0=%.3f\n", output[0], state->cell_state[0]);
    #endif
}

// SNN inference function
void snn_inference(float (*input)[INPUT_SIZE], float *output) {
    HybridNeuron neuron1, neuron2;
    LSTMState lstm_state = {0};

    init_neuron(&neuron1, LSTM_HIDDEN_SIZE, 0.15f, neuron1_scale[0], neuron1_zero_point[0]);
    init_neuron(&neuron2, LSTM_HIDDEN_SIZE, 0.1f, neuron2_scale[0], neuron2_zero_point[0]);

    // Allocate memory for LSTM state
    lstm_state.hidden_state = (float *)calloc(LSTM_HIDDEN_SIZE, sizeof(float));
    lstm_state.cell_state = (float *)calloc(LSTM_HIDDEN_SIZE, sizeof(float));
    if (lstm_state.hidden_state == NULL || lstm_state.cell_state == NULL) {
        fprintf(stderr, "Memory allocation failed for LSTM state\n");
        exit(1);
    }

    float *activity = (float *)calloc(LSTM_HIDDEN_SIZE, sizeof(float));
    if (activity == NULL) {
        fprintf(stderr, "Memory allocation failed for activity\n");
        exit(1);
    }
    const float decay = 0.85f;

    long long start_time = get_time_ms();

    for (int t = 0; t < TIME_STEPS; t++) {
        // Encoder with layer norm and noise injection
        float *encoded = (float *)calloc(HIDDEN_SIZE, sizeof(float));
        if (encoded == NULL) {
            fprintf(stderr, "Memory allocation failed for encoded\n");
            exit(1);
        }

        for (int i = 0; i < HIDDEN_SIZE; i++) {
            float sum = encoder_bias[i];
            for (int j = 0; j < INPUT_SIZE; j++) {
                float noise = NOISE_SCALE * (rand() / (float)RAND_MAX - 0.5f);
                sum += (input[t][j] + noise) * encoder_weight[i * INPUT_SIZE + j];
            }
            sum = (sum - bn_running_mean[i]) / sqrtf(bn_running_var[i] + EPSILON);
            encoded[i] = fmaxf(sum * bn_weight[i] + bn_bias[i], 0.0f);
        }

        // Temporal processing
        float *lstm_out = (float *)calloc(LSTM_HIDDEN_SIZE, sizeof(float));
        if (lstm_out == NULL) {
            fprintf(stderr, "Memory allocation failed for lstm_out\n");
            exit(1);
        }
        lstm_forward(encoded, lstm_out, &lstm_state);

        float *spk1 = (float *)calloc(LSTM_HIDDEN_SIZE, sizeof(float));
        float *spk2 = (float *)calloc(LSTM_HIDDEN_SIZE, sizeof(float));
        if (spk1 == NULL || spk2 == NULL) {
            fprintf(stderr, "Memory allocation failed for spk1/spk2\n");
            exit(1);
        }

        for (int i = 0; i < LSTM_HIDDEN_SIZE; i++) {
            float noise = NOISE_SCALE * (rand() / (float)RAND_MAX - 0.5f);
            spk1[i] = hybrid_neuron_forward(&neuron1, lstm_out[i] + noise, i);
            spk2[i] = hybrid_neuron_forward(&neuron2, spk1[i] + noise, i);
            activity[i] = decay * activity[i] + spk2[i];
        }

        // Sparsity penalty and normalization
        float mean_act = 0.0f;
        for (int i = 0; i < LSTM_HIDDEN_SIZE; i++) mean_act += activity[i];
        mean_act /= LSTM_HIDDEN_SIZE;
        for (int i = 0; i < LSTM_HIDDEN_SIZE; i++) {
            activity[i] -= SPARSITY_WEIGHT * (activity[i] - mean_act);
            activity[i] = fmaxf(activity[i], 0.0f);
        }

        // Display real-time metrics
        if (t % 10 == 0) {
            long long current_time = get_time_ms();
            printf("\n--- Real-Time Metrics ---\n");
            printf("Execution Time: %.2f ms\n", (float)(current_time - start_time));
        }

        // Free temporary arrays
        free(encoded);
        free(lstm_out);
        free(spk1);
        free(spk2);
    }

    // Temperature-scaled decoding with label smoothing
    float max_act = -INFINITY;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = decoder_bias[i];
        for (int j = 0; j < LSTM_HIDDEN_SIZE; j++)
            output[i] += activity[j] * decoder_weight[i * LSTM_HIDDEN_SIZE + j];

        output[i] /= TEMPERATURE;
        if (output[i] > max_act) max_act = output[i];
    }

    // Softmax normalization with label smoothing
    float sum = 0.0f;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = expf(output[i] - max_act) * (1 - LABEL_SMOOTHING) + LABEL_SMOOTHING / OUTPUT_SIZE;
        sum += output[i];
    }
    for (int i = 0; i < OUTPUT_SIZE; i++)
        output[i] /= sum;

    // Cleanup
    free(neuron1.mem); free(neuron2.mem);
    free(lstm_state.hidden_state); free(lstm_state.cell_state);
    free(activity);
}

// Main function
int main() {
    srand(time(NULL));

    // Allocate input and output on the heap
    float (*input)[INPUT_SIZE] = malloc(TIME_STEPS * sizeof(*input));
    float *output = malloc(OUTPUT_SIZE * sizeof(*output));
    if (input == NULL || output == NULL) {
        fprintf(stderr, "Memory allocation failed for input/output\n");
        exit(1);
    }

    // Simulate normalized MNIST input with noise
    for (int t = 0; t < TIME_STEPS; t++)
        for (int i = 0; i < INPUT_SIZE; i++)
            input[t][i] = (rand() % 1000) / 1000.0f * 0.3081 + 0.1307;

    snn_inference(input, output);

    printf("\nClassification Results:\n");
    for (int i = 0; i < OUTPUT_SIZE; i++)
        printf("Class %d: %.4f\n", i, output[i]);

    // Free memory
    free(input);
    free(output);

    return 0;
}
