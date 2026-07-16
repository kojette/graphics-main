//일단 되는 코드. 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "helper_math.h"

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 800
#define WIDTH   1024
#define HEIGHT  1024
#define VOLX 256
#define VOLY 256	
#define VOLZ 225

extern float alphaTable[256];
extern float colorTableR[256];
extern float colorTableG[256];
extern float colorTableB[256];

//tex3d
cudaTextureObject_t volTex = 0;
cudaTextureObject_t sumTex = 0;
cudaArray_t volArray = 0;
cudaArray_t sumArray = 0;
unsigned char* dev_vol = 0, * dev_img = 0; // 볼륨데이터 gpu 메모리 포인터
//unsigned char (*)[256][256] dev_vol = 0; // 볼륨데이터 gpu 메모리 포인터

//전역으로 초기화 해버리기. 
__constant__ float3 at = { 128.0f, 128.0f, 112.0f };
__constant__ float3 up = { 0.0f, 1.0f, 0.0f };


//extern unsigned char vol[VOLZ][VOLY][VOLX];
extern unsigned char MyTexture[HEIGHT][WIDTH][3];
float* dev_alpha = 0;
float* dev_colorR = 0, * dev_colorG = 0, * dev_colorB = 0;
extern unsigned char vol[VOLZ][VOLY][VOLX];
// 만약 호스트의 블록 데이터를 가져온다면 하단처럼 extern 선언 필요
extern unsigned char bM[VOLZ / 8][VOLY / 8][VOLX / 8];


// CUDA 커널 함수: 두 배열을 더함
__global__ void addKernel(int* c, const int* a, const int* b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
__device__ __inline__ int getBlockIdDevice(float3 p) {//아이디 암호화 함수
    int bx = ((int)p.x) >> 3; // BSHIFT = 3
    int by = ((int)p.y) >> 3;
    int bz = ((int)p.z) >> 3;
    return (bx << 10) | (by << 5) | bz;
}
__device__ float GetDensity(cudaTextureObject_t volTex, float3 p) {
    return tex3D<unsigned char>(volTex, p.x, p.y, p.z);



    p.x = fminf(fmaxf(p.x, 0.0f), (float)(VOLX - 2));
    p.y = fminf(fmaxf(p.y, 0.0f), (float)(VOLY - 2));
    p.z = fminf(fmaxf(p.z, 0.0f), (float)(VOLZ - 2));

    int ix = int(p.x);
    int iy = int(p.y);
    int iz = int(p.z);
    float wx = p.x - ix;
    float wy = p.y - iy;
    float wz = p.z - iz;

    // filterMode = Point, normalizedCoords = 0 이므로 정수 좌표 그대로 사용
    float c000 = tex3D<unsigned char>(volTex, ix, iy, iz);
    float c100 = tex3D<unsigned char>(volTex, ix + 1, iy, iz);
    float c010 = tex3D<unsigned char>(volTex, ix, iy + 1, iz);
    float c110 = tex3D<unsigned char>(volTex, ix + 1, iy + 1, iz);
    float c001 = tex3D<unsigned char>(volTex, ix, iy, iz + 1);
    float c101 = tex3D<unsigned char>(volTex, ix + 1, iy, iz + 1);
    float c011 = tex3D<unsigned char>(volTex, ix, iy + 1, iz + 1);
    float c111 = tex3D<unsigned char>(volTex, ix + 1, iy + 1, iz + 1);

    float den =
        c000 * (1 - wx) * (1 - wy) * (1 - wz) +
        c100 * (wx) * (1 - wy) * (1 - wz) +
        c010 * (1 - wx) * (wy) * (1 - wz) +
        c110 * (wx) * (wy) * (1 - wz) +
        c001 * (1 - wx) * (1 - wy) * (wz)+
        c101 * (wx) * (1 - wy) * (wz)+
        c011 * (1 - wx) * (wy) * (wz)+
        c111 * (wx) * (wy) * (wz);

    return den;
}
//아래 코드는 아직 안해봄. 
/*int den = dev_vol[(iz << 16) + (iy << 8) + ix] * (1 - wx) * (1 - wy) * (1 - wz)
    + dev_vol[(iz << 16) + (iy << 8) + (ix + 1)] * (wx) * (1 - wy) * (1 - wz)
    + dev_vol[(iz << 16) + ((iy + 1)<<8) + ix] * (1 - wx) * (wy) * (1 - wz)
    + dev_vol[(iz << 16) + ((iy + 1)<<8) + (ix+1)] * (wx) * (wy) * (1 - wz)
    + dev_vol[((iz+1) << 16) + (iy <<8)+ ix] * (1 - wx) * (1 - wy) * (wz)
    +dev_vol[((iz+1)<< 16) + (iy <<8)+(ix + 1)] * (wx) * (1 - wy) * (wz)
    +dev_vol[((iz+1)<< 16) + ((iy + 1) << 8)+ix] * (1 - wx) * (wy) * (wz)
    +dev_vol[((iz+1)<< 16) + ((iy + 1) << 8)+(ix + 1)] * (wx) * (wy) * (wz);*/

__global__ void mipKernel(cudaTextureObject_t volTex, cudaTextureObject_t sumTex, unsigned char* MyTexture, float3 eye, float* dev_alpha,
    float* dev_colorR, float* dev_colorG, float* dev_colorB) {
    int y = blockIdx.x;
    int x = blockIdx.y * blockDim.x + threadIdx.x; //threadIdx.x;
    //고정값 먼저 만듦. 근데 이거 고정이면 그냥 전역 처리가 나을지도. -> 전역 처리
    //float3 at = make_float3(128.0f, 128.0f, 112.0f);
    //float3 up = make_float3(0.0f, 1.0f, 0.0f);
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
        //sumtable
        float3 block_p = make_float3(p.x / 8.0f, p.y / 8.0f, p.z / 8.0f);
        int blockSum = tex3D<unsigned char>(sumTex, block_p.x + 0.5f, block_p.y + 0.5f, block_p.z + 0.5f);
        if (blockSum == 0) {
            int bid = getBlockIdDevice(p);
            float jump = 0;
            int nextBid;
            do {
                jump += 1.0f;
                if ((t + jump) >= tM) break;
                nextBid = getBlockIdDevice(p + w * jump); //전진한 위치의 ID
            } while (bid == nextBid); // ID가 같으면 같은 블록
            t = t + (jump - step);
            continue;
        }
        /*
        if (p.x < 0 || p.x >= VOLX || p.y < 0 || p.y >= VOLY || p.z < 0 || p.z >= VOLZ) continue;
        // p 위치 볼륨값 읽기
        int ix = (int)p.x, iy = (int)p.y, iz = (int)p.z;
        unsigned char val = vol[iz * VOLY * VOLX + iy * VOLX + ix];
        */ //범위 체크랑 보간을 해주는 함수임. 리턴타입,호스트에서 생성한 텍스터객체, 
        //vol은 그냥 데이터만, voltex는 기능이 탑재도 됨. 
        //그리고 0.5를 추가하는 이유는 중간값(어느 구간의 중간) 자체를 제대로 겟하기 위해서 
        //-> 수치적인 선호에 대한. 0.5. 픽셀 네모의 정중앙을 노렸으나 물리적으로 정확한지는 모르겠음. 
        //unsigned char val = tex3D<unsigned char>(volTex, p.x + 0.5f, p.y + 0.5f, p.z + 0.5f);
        unsigned char val = GetDensity(volTex, p);//float로 받기는 하는데 아래는 전부 int로 쓰넹?
        //if (val > maxVal) maxVal = val;  //mip
        int d = val;  // 밀도값
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

    // 1주차
    // 0. 메모리 할당, 제거 구조화
    // 1. 멈춘 MIP -> 자체 해결; 정확히 말하자면 이게 시점 고정이고 방향도 이상 
    // 2. 움직이는 MIP -> eye값 이용. ray 진행방향 계산.
    // 2.1 eye값을 cumain으로 받아옴
    // 3. ray - box intersection (tmin, tmax)
    // 4. alpha blending 이용해서 볼륨 렌더링 (color, alpha 테이블을 받아옴. dev_alpha, dev_color 메모리로)
    // 4.1 여기서는 sum table은 사용하지 않는다.

    // 2,3주차 휴가
    // 1. cuda Texture라는 기능을 찾아보고, 샘플링을 cuda Texture를 이용해서 구현해보기. AI
    // 1.1 할당 제거 또한 잘 신경 쓸것.
    // 1.2 tex3D 함수 사용하는지 확인
    // 2. sumTable 이용한 VR 구현(이제는 고화질)
    // 3. 이것저것 해보기.
    // 3.1 화면 크기가 더 커지면? 512x512 보다
    // 3.2 실시간 속도 측정. cuda 측정은 cpu 측정과 다르다. 이벤트를 이용한다. AI 검색.
    //MyTexture[(y * WIDTH + x) * 3 + 0] = eye.x;
    //MyTexture[(y * WIDTH + x) * 3 + 1] = eye.y;
    //MyTexture[(y * WIDTH + x) * 3 + 2] = eye.z;
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
    texDesc.filterMode = cudaFilterModePoint; //보간은 수동이 나음. 
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // 원본 볼륨 텍스처 오브젝트 생성
    cudaCreateTextureObject(&volTex, &resDesc, &texDesc, NULL);

    //Sum
    cudaChannelFormatDesc sumChannelDesc = cudaCreateChannelDesc<unsigned char>();
    cudaExtent sumExtent = make_cudaExtent(VOLX / 8, VOLY / 8, VOLZ / 8); // 32, 32, 29

    cudaMalloc3DArray(&sumArray, &sumChannelDesc, sumExtent);

    cudaMemcpy3DParms sumCopyParams = { 0 };
    sumCopyParams.srcPtr = make_cudaPitchedPtr((void*)bM, (VOLX / 8) * sizeof(unsigned char), VOLX / 8, VOLY / 8);
    sumCopyParams.dstArray = sumArray;
    sumCopyParams.extent = sumExtent;
    sumCopyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&sumCopyParams);


    cudaResourceDesc sumResDesc = {};
    sumResDesc.resType = cudaResourceTypeArray;
    sumResDesc.res.array.array = sumArray;

    cudaTextureDesc sumTexDesc = {};
    sumTexDesc.addressMode[0] = cudaAddressModeClamp;
    sumTexDesc.addressMode[1] = cudaAddressModeClamp;
    sumTexDesc.addressMode[2] = cudaAddressModeClamp;
    sumTexDesc.filterMode = cudaFilterModePoint; // 블록 검사는 보간 없이 딱딱 끊어지게 Point 모드//보간 모드는 어떤 사유로 안됨. 
    sumTexDesc.readMode = cudaReadModeElementType;
    sumTexDesc.normalizedCoords = 0;
    cudaCreateTextureObject(&sumTex, &sumResDesc, &sumTexDesc, NULL);


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
    // 메모리 해제는 여기서 한다.
    //cudaFree(dev_vol);
    //텍스터 오브젝 파괴하고 배열 해제
    if (volTex) cudaDestroyTextureObject(volTex);
    if (sumTex) cudaDestroyTextureObject(sumTex);
    if (volArray) cudaFreeArray(volArray);
    if (sumArray) cudaFreeArray(sumArray);
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
    //grid.x = y, grid.y = x방향을 256개씩 쪼갠 블록? 개수
    mipKernel << <grid, block >> > (volTex, sumTex, dev_img, eye, dev_alpha, dev_colorR, dev_colorG, dev_colorB);
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
