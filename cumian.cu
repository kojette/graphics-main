#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "helper_math.h"

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 800
#define WIDTH   2048
#define HEIGHT  2048
#define VOLX 256
#define VOLY 256	
#define VOLZ 225

extern float alphaTable[256];
extern float colorTableR[256];
extern float colorTableG[256];
extern float colorTableB[256];

//tex3d
cudaTextureObject_t volTex = 0;
cudaArray_t volArray = 0;

unsigned char* dev_vol = 0, * dev_img = 0; // 볼륨데이터 gpu 메모리 포인터

//전역으로 초기화 해버리기. 
__constant__ float3 at = { 128.0f, 128.0f, 112.0f };
__constant__ float3 up = { 0.0f, 1.0f, 0.0f };

//extern unsigned char vol[VOLZ][VOLY][VOLX];
extern unsigned char MyTexture[HEIGHT][WIDTH][3];
float* dev_alpha = 0;
float* dev_colorR = 0, * dev_colorG = 0, * dev_colorB = 0;
extern unsigned char vol[VOLZ][VOLY][VOLX];



// CUDA 커널 함수: 두 배열을 더함
__global__ void addKernel(int* c, const int* a, const int* b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
__global__ void mipKernel(cudaTextureObject_t volTex, unsigned char* MyTexture, float3 eye, float* dev_alpha,
    float* dev_colorR, float* dev_colorG, float* dev_colorB) {
    int y = blockIdx.x;
    int x = blockIdx.y * blockDim.x + threadIdx.x; //threadIdx.x;
    //카메라 축 계산
    float3 w = normalize(at - eye);
    float3 u = normalize(cross(up, w));
    float3 v = normalize(cross(w, u));
    //레이캐스팅 
    const float supersampling = 0.5f * (512.0f / WIDTH);
    float3 RS = eye + u * (x - WIDTH * 0.5f) * supersampling + v * (y - HEIGHT * 0.5f) * supersampling;

    float t1, t2; // 한 구간
    t1 = -RS.x / w.x;
    t2 = (255 - RS.x) / w.x;
    float xm = fminf(t1, t2);
    float xM = fmaxf(t1, t2);

    t1 = -RS.y / w.y;
    t2 = (255 - RS.y) / w.y;
    float ym = fminf(t1, t2);
    float yM = fmaxf(t1, t2);

    t1 = -RS.z / w.z;
    t2 = (224 - RS.z) / w.z;
    float zm = fminf(t1, t2);
    float zM = fmaxf(t1, t2);
    float tm = fmaxf(fmaxf(xm, ym), zm);
    float tM = fminf(fminf(xM, yM), zM);
    if (tm > tM) return;

    const float step = 0.5;
    float maxVal = 0.0f;//MIP관련으로 
    float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f;
    float a_sum = 0.0f;
    for (float t = tm; t < tM; t = t + step) { //광선
        float3 p = RS + w * t;

        //쿠다텍스처는 좌표가 경계를 의미하지 않아서 오프셋 처리 해야 결과 유지됨. 
        float val = tex3D<float>(volTex, p.x + 0.5f, p.y + 0.5f, p.z + 0.5f);//(volTex, p.x, p.y, p.z); 
        int d = (int)(val * 255.0f);//원래 밀도 스케일(0~255)로 변환함. 
        float alpha = dev_alpha[d];
        if (alpha == 0.0f) continue;
        float r = dev_colorR[d];
        float g = dev_colorG[d];
        float b = dev_colorB[d];
        r_sum += (1.0f - a_sum) * r * alpha;
        g_sum += (1.0f - a_sum) * g * alpha;
        b_sum += (1.0f - a_sum) * b * alpha;
        a_sum += (1.0f - a_sum) * alpha;
        if (a_sum > 0.99f) break;
    }
    MyTexture[(y * WIDTH + x) * 3 + 0] = (unsigned char)(r_sum * 255);
    MyTexture[(y * WIDTH + x) * 3 + 1] = (unsigned char)(g_sum * 255);
    MyTexture[(y * WIDTH + x) * 3 + 2] = (unsigned char)(b_sum * 255);
}

extern "C" int cuInit() {
    // 1. 출력 이미지 버퍼 할당
    cudaError_t err;
    err = cudaMalloc((void**)&dev_img, HEIGHT * WIDTH * 3 * sizeof(unsigned char));
    if (err != cudaSuccess) return -1;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
    cudaExtent extent = make_cudaExtent(VOLX, VOLY, VOLZ);

    // 3D CUDA Array 할당
    cudaMalloc3DArray(&volArray, &channelDesc, extent);

    // Host -> Device 3D Array 복사 세팅
    cudaMemcpy3DParms copyParams = { 0 };
    copyParams.srcPtr = make_cudaPitchedPtr((void*)vol, VOLX * sizeof(unsigned char), VOLX, VOLY);
    copyParams.dstArray = volArray;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);

    // 텍스처 지정자(Resource Desc) 설정
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = volArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;//이거 char를 0~1 사이 float으로. 이거 때문에 안된 걸지도?
    texDesc.normalizedCoords = 0;

    // 원본 볼륨 텍스처 오브젝트 생성
    cudaCreateTextureObject(&volTex, &resDesc, &texDesc, NULL);

    cudaMalloc((void**)&dev_alpha, 256 * sizeof(float));
    cudaMalloc((void**)&dev_colorR, 256 * sizeof(float));
    cudaMalloc((void**)&dev_colorG, 256 * sizeof(float));
    cudaMalloc((void**)&dev_colorB, 256 * sizeof(float));
    cudaMemcpy(dev_alpha, alphaTable, 256 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_colorR, colorTableR, 256 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_colorG, colorTableG, 256 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_colorB, colorTableB, 256 * sizeof(float), cudaMemcpyHostToDevice);

    return 0;
}
extern "C" int cuFree() {
    // 메모리 해제는 여기서 한다. cudaFree(dev_vol);
    //텍스터 오브젝 파괴하고 배열 해제
    if (volTex) cudaDestroyTextureObject(volTex);
    if (volArray) cudaFreeArray(volArray);
    cudaFree(dev_img);
    cudaFree(dev_alpha);
    cudaFree(dev_colorR);
    cudaFree(dev_colorG);
    cudaFree(dev_colorB);
    return 0;

}

extern "C" int cumain(float ex, float ey, float ez) { // 메모리할당은 cuInit에서 하고, 커널 실행은 cumain에서 한다.
    // 데이터 복사
    //cudaMemcpy(dev_vol, vol, VOLX * VOLY * VOLZ * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // 커널 실행
    float3 eye = make_float3(ex, ey, ez);

    int threadsPerBlock = 256;
    dim3 block(threadsPerBlock);
    dim3 grid(HEIGHT, (WIDTH + threadsPerBlock - 1) / threadsPerBlock);
    //grid.x = y, grid.y = x방향을 256개씩 쪼갠 블록? 개수(스레드 개수땜)
    mipKernel << <grid, block >> > (volTex, dev_img, eye, dev_alpha, dev_colorR, dev_colorG, dev_colorB);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop); // 커널 끝날 때까지 대기
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("kernel time: %f ms\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // 결과 복사//
    cudaMemcpy(MyTexture, dev_img, HEIGHT * WIDTH * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    return 0;
}