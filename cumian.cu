// cumain_v5_bench.cu
// v4 -> v5 : 법선 3버전을 컴파일 타임 스위치(NORMAL_VERSION)로 분리 — 순수 실행시간 측정용
//            프레임 시간 통계(워밍업 제외, 평균/표준편차/최소/최대) 추가
//            프리인티그레이션 테이블의 GPU 생성 커널 추가 (CPU판과 대조용)

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>          // 추가: 표준편차 계산의 sqrtf 때문
#include "helper_math.h"

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 800
#define WIDTH   2048
#define HEIGHT  2048
#define VOLX 256
#define VOLY 256
#define VOLZ 225
#define TABLE_SIZE 256

// 주의: main.cpp 에도 같은 이름의 #define 이 있다. 두 파일의 값이 다르면
//       테이블은 A 길이로 만들고 광선은 B 길이로 걷는 사고가 난다. 반드시 같게 둘 것.
#define DEFAULT_SEGMENT_LENGTH 0.5f

//====== 측정용 스위치 =========================================================
// 1 = 앞 법선만, 2 = 뒤 법선만, 3 = 앞뒤 보간
// 왜 #if 인가: 런타임 if 로 고르면 세 버전이 전부 컴파일되어 레지스터 사용량이
//             같아지고, 안 쓰는 tex3D 도 스케줄에 남을 수 있다. 그러면 "순수
//             실행시간"이 아니라 "셋 다 계산한 시간"을 재게 된다. 값을 바꿔
//             세 번 빌드하는 것이 가장 원시적이고 가장 정확하다.
#define NORMAL_VERSION 4 
#define NORMAL_POS 0.5f        // 버전 3에서 뒤 법선의 비중 (0 이면 앞, 1 이면 뒤)

#define WARMUP_FRAMES 120        // 첫 프레임들은 캐시/JIT 때문에 느리다. 버린다.
#define REPORT_FRAMES 200       // 이만큼 모아서 한 번 통계 출력
//=============================================================================

// GPU측 포인터
float* dev_preAlpha = 0;
float* dev_preColorR = 0, * dev_preColorG = 0, * dev_preColorB = 0;

// 추가: GPU가 직접 만든 테이블을 담을 곳 (렌더링용 dev_preAlpha 와 섞이면 대조가 안 되므로 분리)
float* dev_gpuAlpha = 0;
float* dev_gpuColorR = 0, * dev_gpuColorG = 0, * dev_gpuColorB = 0;

extern float alphaTable[256];
extern float colorTableR[256];
extern float colorTableG[256];
extern float colorTableB[256];

extern float preAlpha[TABLE_SIZE * TABLE_SIZE];
extern float preColorR[TABLE_SIZE * TABLE_SIZE];
extern float preColorG[TABLE_SIZE * TABLE_SIZE];
extern float preColorB[TABLE_SIZE * TABLE_SIZE];

// 추가: GPU 결과를 받아갈 호스트 배열 (main.cpp 에 실체가 있다)
extern float gpuAlpha[TABLE_SIZE * TABLE_SIZE];
extern float gpuColorR[TABLE_SIZE * TABLE_SIZE];
extern float gpuColorG[TABLE_SIZE * TABLE_SIZE];
extern float gpuColorB[TABLE_SIZE * TABLE_SIZE];

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

// 추가: 법선만 뽑는 함수. 왜 쪼갰나 — 같은 위치에서 두 번 뽑을 수 있어야
//       버전 3(앞뒤 보간)이 성립하고, tex3D 호출 수를 눈으로 셀 수 있다.
//       한 번 부를 때 tex3D 6회.
__device__ float3 GetNormal(cudaTextureObject_t volTex, float3 p)
{
    // 중앙차분
    float dx = tex3D<float>(volTex, p.x + 1.5f, p.y + 0.5f, p.z + 0.5f)
        - tex3D<float>(volTex, p.x - 0.5f, p.y + 0.5f, p.z + 0.5f);
    float dy = tex3D<float>(volTex, p.x + 0.5f, p.y + 1.5f, p.z + 0.5f)
        - tex3D<float>(volTex, p.x + 0.5f, p.y - 0.5f, p.z + 0.5f);
    float dz = tex3D<float>(volTex, p.x + 0.5f, p.y + 0.5f, p.z + 1.5f)
        - tex3D<float>(volTex, p.x + 0.5f, p.y + 0.5f, p.z - 0.5f);

    return make_float3(dx, dy, dz);
}

// 수정: 위치가 아니라 법선을 받는다. 정규화는 여기서 한 번만.
__device__ float lighting(float3 N, float3 w)
{
    float len = length(N);
    if (len < 1e-6f) return 1.0f;   // 평평한 곳은 명암 없음(0으로 나누기 방지)
    N = N / len;

    float3 L = w;                    // 광원 = 시선 방향 (헤드라이트)
    float3 V = w;
    float3 H = normalize(L + V);

    float NL = fabsf(dot(N, L));
    float NH = fabsf(dot(N, H));

    const float Ia = 0.25f, Id = 0.5f, Is = 0.9f;
    return Ia + Id * NL + Is * __powf(NH, 30.0f);
}

//=== 추가: 프리인티그레이션 테이블을 GPU에서 만드는 커널 ======================
// 스레드 하나가 표의 칸 하나(sf, sb)를 맡는다. 칸끼리 서로를 안 보므로
// 통째로 병렬이 된다 — 이게 이 표가 GPU에 어울리는 이유.
// CPU 판(BuildPreIntegrationTable)과 계산 순서를 한 줄도 다르게 쓰지 않았다.
// 다르게 쓰면 결과가 달라져서 "GPU가 맞게 만들었나"를 대조할 수 없다.
__global__ void buildPreIntKernel(float* outAlpha, float* outR, float* outG, float* outB,
    float* devAlphaTable, float* devColorR, float* devColorG, float* devColorB)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TABLE_SIZE * TABLE_SIZE) return;

    int sf = idx >> 8;        // TABLE_SIZE 가 256 이라 8비트 시프트 = 나눗셈
    int sb = idx & 0xFF;      // 하위 8비트 = 나머지

    float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f, a_sum = 0.0f;

    int n = (sb > sf) ? (sb - sf) : (sf - sb);
    if (n == 0) {
        float a = devAlphaTable[sf];
        a = 1.0f - powf(1.0f - a, DEFAULT_SEGMENT_LENGTH);
        a_sum = a;
        r_sum = devColorR[sf] * a;
        g_sum = devColorG[sf] * a;
        b_sum = devColorB[sf] * a;
    }
    else {
        float width = (float)n;
        float inv_width = DEFAULT_SEGMENT_LENGTH / width;
        int dir = (sb > sf) ? 1 : -1;

        for (int i = sf; i != sb; i += dir) {
            float alpha = devAlphaTable[i];
            if (alpha > 0.0f) {
                float ca = 1.0f - powf(1.0f - alpha, inv_width);

                r_sum += (1.0f - a_sum) * devColorR[i] * ca;
                g_sum += (1.0f - a_sum) * devColorG[i] * ca;
                b_sum += (1.0f - a_sum) * devColorB[i] * ca;
                a_sum += (1.0f - a_sum) * ca;

                if (a_sum > 0.99f) break;
            }
        }
    }

    outAlpha[idx] = a_sum;
    outR[idx] = r_sum;
    outG[idx] = g_sum;
    outB[idx] = b_sum;
}
//=============================================================================

__global__ void mipKernel(cudaTextureObject_t volTex, unsigned char* MyTexture, float3 eye,
    float* dev_preAlpha, float* dev_preColorR, float* dev_preColorG, float* dev_preColorB) {
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

    const float step = DEFAULT_SEGMENT_LENGTH;
    float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f, a_sum = 0.0f;
    //최초 1회 front만 읽고 이후에는 back을 사용하기 위해서 아래 계산을 시행.
    int sf = (int)roundf(tex3D<float>(volTex, RS.x + w.x * tm + 0.5f, RS.y + w.y * tm + 0.5f, RS.z + w.z * tm + 0.5f) * (float)(TABLE_SIZE - 1));
    for (float t = tm; t < tM - step; t += step) { //광선
        float3 pf = RS + w * t;         // 추가: 구간의 앞점. sf 를 읽은 그 자리다.
        float3 p = RS + w * (t + step); //뒤에

        //쿠다텍스처는 좌표가 경계를 의미하지 않아서 오프셋 처리 해야 결과 유지됨.
        float val = tex3D<float>(volTex, p.x + 0.5f, p.y + 0.5f, p.z + 0.5f);
        int sb = (int)roundf(val * (float)(TABLE_SIZE - 1));

        int d = (int)(sf * TABLE_SIZE + sb);//원래 밀도 스케일(0~255)로 변환하여 id get
        float alpha = dev_preAlpha[d];

        if (alpha > 0.0f) {
            // 버전에 따라 실제로 계산하는 법선의 개수가 달라진다.
            // 이것이 "순수 실행시간 차이"의 정체다. tex3D 6회 vs 12회.
#if   NORMAL_VERSION == 1
            float3 N = GetNormal(volTex, pf);                    // 앞만  : tex3D 6회
#elif NORMAL_VERSION == 2
            float3 N = GetNormal(volTex, p);                     // 뒤만  : tex3D 6회
#elif NORMAL_VERSION == 3
            float3 Nf = GetNormal(volTex, pf);                   // 앞뒤  : tex3D 12회
            float3 Nb = GetNormal(volTex, p);
            float3 N = Nf * (1.0f - NORMAL_POS) + Nb * NORMAL_POS; // if 없이 곱셈 보간
#else
            float3 Nf = GetNormal(volTex, pf);                   // 신뢰도 가중 : tex3D 12회
            float3 Nb = GetNormal(volTex, p);
            float lf = length(Nf);              // 앞 법선의 크기 = 앞이 얼마나 믿을 만한가
            float lb = length(Nb);              // 뒤 법선의 크기 = 뒤가 얼마나 믿을 만한가
            float nw = lb / (lf + lb + 1e-6f);  // 뒤가 가져갈 표. 1e-6 은 0으로 나누기 방지
            float3 N = Nf * (1.0f - nw) + Nb * nw;
#endif
            float light = lighting(N, w);//조명 추가
            float r = dev_preColorR[d] * light;
            float g = dev_preColorG[d] * light;
            float b = dev_preColorB[d] * light;

            r_sum += (1.0f - a_sum) * r;
            g_sum += (1.0f - a_sum) * g;
            b_sum += (1.0f - a_sum) * b;
            a_sum += (1.0f - a_sum) * alpha;
        }
        if (a_sum > 0.99f) break;
        sf = sb;//다음 회차에 쓰려구

    }
    r_sum = fminf(fmaxf(r_sum, 0.0f), 1.0f);
    g_sum = fminf(fmaxf(g_sum, 0.0f), 1.0f);
    b_sum = fminf(fmaxf(b_sum, 0.0f), 1.0f);
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
    texDesc.readMode = cudaReadModeNormalizedFloat;
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


    cudaMalloc((void**)&dev_preAlpha, TABLE_SIZE * TABLE_SIZE * sizeof(float));
    cudaMalloc((void**)&dev_preColorR, TABLE_SIZE * TABLE_SIZE * sizeof(float));
    cudaMalloc((void**)&dev_preColorG, TABLE_SIZE * TABLE_SIZE * sizeof(float));
    cudaMalloc((void**)&dev_preColorB, TABLE_SIZE * TABLE_SIZE * sizeof(float));
    cudaMemcpy(dev_preAlpha, preAlpha, TABLE_SIZE * TABLE_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_preColorR, preColorR, TABLE_SIZE * TABLE_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_preColorG, preColorG, TABLE_SIZE * TABLE_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_preColorB, preColorB, TABLE_SIZE * TABLE_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // 추가: GPU 생성 실험용 버퍼. 렌더링에는 쓰지 않는다(대조 전용).
    cudaMalloc((void**)&dev_gpuAlpha, TABLE_SIZE * TABLE_SIZE * sizeof(float));
    cudaMalloc((void**)&dev_gpuColorR, TABLE_SIZE * TABLE_SIZE * sizeof(float));
    cudaMalloc((void**)&dev_gpuColorG, TABLE_SIZE * TABLE_SIZE * sizeof(float));
    cudaMalloc((void**)&dev_gpuColorB, TABLE_SIZE * TABLE_SIZE * sizeof(float));

    return 0;
}

//=== 추가: 테이블 GPU 생성 1회 실행 + 시간 반환 ===============================
// 반환값 = 커널 시간(ms). 복사 시간은 따로 인쇄한다.
// 왜 나누나: "GPU가 빠르다"는 주장은 복사 비용을 감추면 절반짜리다.
extern "C" float cuBuildPreIntegrationTable() {
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    int total = TABLE_SIZE * TABLE_SIZE;   // 65536
    int threads = 256;
    int blocks = total / threads;          // 256 개, 나머지 없음

        // 워밍업 1회 — 첫 커널 실행에는 CUDA 시동 비용이 섞여 든다. 그건 버린다.
    buildPreIntKernel << <blocks, threads >> > (dev_gpuAlpha, dev_gpuColorR, dev_gpuColorG, dev_gpuColorB,
        dev_alpha, dev_colorR, dev_colorG, dev_colorB);
    cudaDeviceSynchronize();

    float ms = 0, sum = 0.0f, lo = 1e9f;
    for (int i = 0; i < 5; i++) {          // CPU 쪽 5회와 조건을 맞춘다
        cudaEventRecord(s);
        buildPreIntKernel << <blocks, threads >> > (dev_gpuAlpha, dev_gpuColorR, dev_gpuColorG, dev_gpuColorB,
            dev_alpha, dev_colorR, dev_colorG, dev_colorB); 
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        cudaEventElapsedTime(&ms, s, e);
        printf("[GPU table] run %d : %.4f ms\n", i, ms);
        sum = sum + ms;
        if (ms < lo) lo = ms;
    }
    printf("[GPU table] mean %.4f ms, min %.4f ms\n", sum / 5.0f, lo);
    ms = lo;                               // 아래 printf 와 return 이 최솟값을 쓰도록

    cudaEventRecord(s);
    cudaMemcpy(gpuAlpha, dev_gpuAlpha, total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpuColorR, dev_gpuColorR, total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpuColorG, dev_gpuColorG, total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpuColorB, dev_gpuColorB, total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float msCopy = 0;
    cudaEventElapsedTime(&msCopy, s, e);
    printf("[GPU table] kernel %.4f ms, copyback %.4f ms\n", ms, msCopy);

    cudaEventDestroy(s);
    cudaEventDestroy(e);
    return ms;
}
//=============================================================================

extern "C" int cuFree() {
    if (volTex) cudaDestroyTextureObject(volTex);
    if (volArray) cudaFreeArray(volArray);
    cudaFree(dev_img);
    cudaFree(dev_alpha);
    cudaFree(dev_colorR);
    cudaFree(dev_colorG);
    cudaFree(dev_colorB);
    cudaFree(dev_gpuAlpha);       // 추가
    cudaFree(dev_gpuColorR);      // 추가
    cudaFree(dev_gpuColorG);      // 추가
    cudaFree(dev_gpuColorB);      // 추가
    return 0;

}

extern "C" int cumain(float ex, float ey, float ez) { // 메모리할당은 cuInit에서 하고, 커널 실행은 cumain에서 한다.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // 커널 실행
    float3 eye = make_float3(ex, ey, ez);

    int threadsPerBlock = 256;
    dim3 block(threadsPerBlock);
    dim3 grid(HEIGHT, (WIDTH + threadsPerBlock - 1) / threadsPerBlock);
    mipKernel << <grid, block >> > (volTex, dev_img, eye, dev_preAlpha, dev_preColorR, dev_preColorG, dev_preColorB);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop); // 커널 끝날 때까지 대기
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    //=== 추가: 프레임 시간 통계 ==============================================
    // 한 프레임 숫자 하나로는 아무것도 말할 수 없다. 워밍업을 버리고 모아서
    // 평균과 표준편차를 낸다. 표준편차가 크면 그 평균은 믿을 값이 아니다.
    static int seen = 0;          // 지금까지 본 프레임 수 (워밍업 포함)
    static int n = 0;             // 통계에 넣은 프레임 수
    static float sum = 0.0f;      // 합
    static float sumSq = 0.0f;    // 제곱합 (표준편차용)
    static float lo = 1e9f;       // 최소
    static float hi = 0.0f;       // 최대

    seen = seen + 1;
    if (seen > WARMUP_FRAMES) {
        n = n + 1;
        sum = sum + ms;
        sumSq = sumSq + ms * ms;
        lo = fminf(lo, ms);
        hi = fmaxf(hi, ms);

        if (n == REPORT_FRAMES) {
            float mean = sum / (float)n;
            float var = sumSq / (float)n - mean * mean;   // E[x^2] - E[x]^2
            if (var < 0.0f) var = 0.0f;                   // 부동소수 오차로 음수가 될 수 있음
            printf("[ver %d] n=%d  mean %.4f ms  sd %.4f  min %.4f  max %.4f\n",
                NORMAL_VERSION, n, mean, sqrtf(var), lo, hi);
            n = 0; sum = 0.0f; sumSq = 0.0f; lo = 1e9f; hi = 0.0f;
        }
    }
    //=======================================================================

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // 결과 복사//
    cudaMemcpy(MyTexture, dev_img, HEIGHT * WIDTH * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    return 0;
}