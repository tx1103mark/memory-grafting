#include <cuda_runtime_api.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static volatile sig_atomic_t running = 1;
static void stop(int signum) { (void)signum; running = 0; }

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s DEVICE RESERVE_GIB\n", argv[0]);
        return 2;
    }
    int device = atoi(argv[1]);
    size_t bytes = (size_t)atoll(argv[2]) << 30;
    void *allocation = NULL;
    cudaError_t err = cudaSetDevice(device);
    if (err == cudaSuccess) err = cudaMalloc(&allocation, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU %d reservation failed: %s\n", device, cudaGetErrorString(err));
        return 1;
    }
    signal(SIGTERM, stop);
    signal(SIGINT, stop);
    printf("reserved %s GiB on GPU %d, pid=%d\n", argv[2], device, getpid());
    fflush(stdout);
    while (running) pause();
    cudaFree(allocation);
    return 0;
}
