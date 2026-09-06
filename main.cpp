// main_v5_bench.cpp
// v4 -> v5 : 테이블 CPU 생성 시간을 5회 반복 측정(평균/최소)으로 바꿈
//            GPU 생성 테이블을 받아 CPU판과 최대오차 대조하는 CompareTables 추가
//            측정 중 카메라를 고정하는 스위치(FIX_CAMERA) 추가 — 변인 하나 원칙

#include <GL/glut.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>

#include <vector>
#include <algorithm>
// 시간 측정등 고성능 함수
#include <chrono>
using namespace std;

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 800
#define WIDTH   2048
#define HEIGHT  2048
#define VOLX 256
#define VOLY 256
#define VOLZ 225
const int BSIZE = 8;
const int BSHIFT = 3;

//====== 측정용 스위치 =========================================================
// 1 이면 카메라를 한 자리에 못박는다.
// 왜 필요한가: MyDisplay 는 매 프레임 eye 를 움직인다. 시점이 바뀌면 광선이
//             볼륨을 지나는 길이가 바뀌고, 조기종료 시점도 바뀐다. 그 상태로
//             버전 1/2/3 시간을 비교하면 무엇이 시간을 바꿨는지 알 수 없다.
#define FIX_CAMERA 1
#define TABLE_BENCH_RUNS 5   // CPU 테이블 생성을 몇 번 반복해 잴 것인가
//=============================================================================

unsigned char ImageBuf[HEIGHT][WIDTH];
unsigned char MyTexture[HEIGHT][WIDTH][3];
unsigned char vol[VOLZ][VOLY][VOLX];

//수정 이전
const int BZ_COUNT = VOLZ / BSIZE + (VOLZ % BSIZE != 0); // 29
const int BY_COUNT = VOLY / BSIZE + (VOLY % BSIZE != 0); // 32
const int BX_COUNT = VOLX / BSIZE + (VOLX % BSIZE != 0); // 32
unsigned char bM[BZ_COUNT][BY_COUNT][BX_COUNT];
unsigned char bm[BZ_COUNT][BY_COUNT][BX_COUNT];

float alphaTable[256];
float sumTable[256];
float colorTableR[256];
float colorTableG[256];
float colorTableB[256];

//pre-integration
// 주의: cumain.cu 에도 같은 이름의 #define 이 있다. 두 값이 다르면 테이블과
//       광선이 서로 다른 구간 길이를 쓴다. 반드시 같은 값으로 둘 것.
#define DEFAULT_SEGMENT_LENGTH 0.5f  // 구간 길이. 커널의 step 과 같은 값 (0.5 -> 2.0)
#define TABLE_SIZE 256
// 좌표축이 (sf, sb) 두 개인 2D 테이블. CPU에서 만든 뒤 GPU로 1회 업로드.
// 배열은 1차원으로
float preAlpha[TABLE_SIZE * TABLE_SIZE];
float preColorR[TABLE_SIZE * TABLE_SIZE];
float preColorG[TABLE_SIZE * TABLE_SIZE];
float preColorB[TABLE_SIZE * TABLE_SIZE];

// 추가: GPU 가 만든 표를 받아둘 곳. 렌더링에는 쓰지 않고 대조에만 쓴다.
float gpuAlpha[TABLE_SIZE * TABLE_SIZE];
float gpuColorR[TABLE_SIZE * TABLE_SIZE];
float gpuColorG[TABLE_SIZE * TABLE_SIZE];
float gpuColorB[TABLE_SIZE * TABLE_SIZE];

//구조체----------------------------------------------------------------
struct alphaPoint {
	int x;//density
	float y;//alpha
};

class AlphaTable {
public:
	vector<alphaPoint> alphas;
	void AddPoint(int x, float y) {
		alphas.push_back({ x, y });
	}
	void MakeAlphaTable(float alphaTable[256]) {
		if (alphas.size() < 2) return;//만약 점 안 찍을 경우 예외가 없을 것을 대비

		//x 기준 정렬
		sort(alphas.begin(), alphas.end(), [](const alphaPoint& a, const alphaPoint& b) {
			return a.x < b.x;
			});
		for (int i = 0; i < alphas.size() - 1; i++) {//구간 별
			int x0 = alphas[i].x;
			int x1 = alphas[i + 1].x;
			float y0 = alphas[i].y;
			float y1 = alphas[i + 1].y;

			float _len = 1 / float(x1 - x0);
			for (int j = x0; j <= x1; j++) {//한 구간 내
				float a = (y0 * (x1 - j) + y1 * (j - x0)) * _len;
				if (a < 0) a = 0;
				else if (a > 1) a = 1;
				alphaTable[j] = a;
			}
		}
	}
};

//color버전
struct colorPoint {
	int x;      // density
	float r;    // red
	float g;    // green
	float b;    // blue
};

class ColorTable {
public:
	vector<colorPoint> colors;

	void AddPoint(int x, float r, float g, float b) {
		colors.push_back({ x, r, g, b });
	}

	void MakeColorTable(float colorTableR[256],
		float colorTableG[256],
		float colorTableB[256]) {
		if (colors.size() < 2) return; //점 부족 예외 처리

		//x 기준 정렬
		sort(colors.begin(), colors.end(),
			[](const colorPoint& a, const colorPoint& b) {
				return a.x < b.x;
			});

		for (int i = 0; i < colors.size() - 1; i++) {
			//가독성 정리
			int x0 = colors[i].x;
			int x1 = colors[i + 1].x;

			float r0 = colors[i].r;
			float g0 = colors[i].g;
			float b0 = colors[i].b;

			float r1 = colors[i + 1].r;
			float g1 = colors[i + 1].g;
			float b1 = colors[i + 1].b;

			for (int j = x0; j <= x1; j++) {
				float t = float(j - x0) / float(x1 - x0);//나눗셈 연산 비싸니까 한번으로

				float r = r0 * (1 - t) + r1 * t;
				float g = g0 * (1 - t) + g1 * t;
				float b = b0 * (1 - t) + b1 * t;

				colorTableR[j] = glm::clamp(r, 0.0f, 1.0f);//보여주신 함수?
				colorTableG[j] = glm::clamp(g, 0.0f, 1.0f);
				colorTableB[j] = glm::clamp(b, 0.0f, 1.0f);
			}
		}
	}
};
//함수들----------------------------------------------------------------
void FileRead()
{
	std::ifstream myfile;
	myfile.open("bighead.den", std::ios::in | std::ios::binary);
	if (!myfile.is_open()) {
		std::cout << "file error";
	}
	myfile.read((char*)vol, VOLZ * VOLY * VOLX);
	myfile.close();
}

inline bool isOutside(const glm::vec3& p) {//범위 처리 따라, 알파 컬러에서는 불필요
	if (p.x >= VOLX || p.x < 0 ||
		p.y >= VOLY || p.y < 0 ||
		p.z >= VOLZ || p.z < 0) return true;
	else
		return false;
}

void GenBlocks() { //수정A-2: 29, 32, 32에서 각각 B~_COUNT
	for (int bz = 0; bz < BZ_COUNT; bz++) // BZ = 28
		for (int by = 0; by < BY_COUNT; by++)
			for (int bx = 0; bx < BX_COUNT; bx++) { // 각 블록에 대해서
				unsigned char max_value = 0, min_value = 255;
				// 최대값을 추출해서 //(개선+; 경계값 추가)
				for (int z = bz * BSIZE; z <= __min(bz * BSIZE + BSIZE, VOLZ - 1); z++) {
					for (int y = by * BSIZE; y <= __min(by * BSIZE + BSIZE, VOLY - 1); y++) {
						for (int x = bx * BSIZE; x <= __min(bx * BSIZE + BSIZE, VOLX - 1); x++) {
							max_value = __max(vol[z][y][x], max_value);
							min_value = __min(vol[z][y][x], min_value);
						}
					}
				}
				// 저장한다.
				bM[bz][by][bx] = max_value;
				bm[bz][by][bx] = min_value;
			}
	printf("max = %d, min = %d \n", bM[14][16][16], bm[14][16][16]);
}

int GetDensity(glm::vec3 p) {
	int ix = int(p.x); // 4.8 ->  4
	int iy = int(p.y); // 4.8 ->  4
	int iz = int(p.z); // 4.8 ->  4
	float wx = p.x - ix;
	float wy = p.y - iy;
	float wz = p.z - iz;
	//000	001 010 011 100 101 110 111
	int den = vol[iz][iy][ix] * (1 - wx) * (1 - wy) * (1 - wz)
		+ vol[iz][iy][ix + 1] * (wx) * (1 - wy) * (1 - wz)
		+ vol[iz][iy + 1][ix] * (1 - wx) * (wy) * (1 - wz)
		+ vol[iz][iy + 1][ix + 1] * (wx) * (wy) * (1 - wz)
		+ vol[iz + 1][iy][ix] * (1 - wx) * (1 - wy) * (wz)
		+vol[iz + 1][iy][ix + 1] * (wx) * (1 - wy) * (wz)
		+vol[iz + 1][iy + 1][ix] * (1 - wx) * (wy) * (wz)
		+vol[iz + 1][iy + 1][ix + 1] * (wx) * (wy) * (wz);
	return den;
}

void InitTables() {
	AlphaTable mat;
	mat.AddPoint(0, 0.0f);      //공기&연조직 비가시
	mat.AddPoint(90, 0.0f);
	mat.AddPoint(98, 0.85f);    //여기부터 8단위로 진동
	mat.AddPoint(106, 0.0f);
	mat.AddPoint(114, 0.85f);
	mat.AddPoint(122, 0.0f);
	mat.AddPoint(130, 0.85f);
	mat.AddPoint(138, 0.0f);
	mat.AddPoint(146, 0.85f);
	mat.AddPoint(154, 0.0f);
	mat.AddPoint(162, 0.85f);
	mat.AddPoint(170, 0.0f);
	mat.AddPoint(178, 0.85f);
	mat.AddPoint(186, 0.0f);
	mat.AddPoint(194, 0.85f);
	mat.AddPoint(202, 0.0f);
	mat.AddPoint(210, 0.85f);
	mat.AddPoint(218, 0.0f);
	mat.AddPoint(226, 0.85f);
	mat.AddPoint(234, 0.0f);
	mat.AddPoint(242, 0.85f);
	mat.AddPoint(250, 0.0f);
	mat.AddPoint(255, 0.0f);
	mat.MakeAlphaTable(alphaTable);

	sumTable[0] = alphaTable[0];
	for (int i = 1; i < 256; i++) {
		sumTable[i] = sumTable[i - 1] + alphaTable[i];
	} //in : alphaTable, out : sumTable

	ColorTable ct;
	ct.AddPoint(0, 0.0f, 0.0f, 0.0f);
	ct.AddPoint(90, 0.1f, 0.2f, 0.55f);
	ct.AddPoint(106, 0.95f, 0.3f, 0.2f);   //봉우리마다 색이 바뀌도록
	ct.AddPoint(122, 0.2f, 0.9f, 0.4f);
	ct.AddPoint(138, 0.3f, 0.4f, 1.0f);
	ct.AddPoint(154, 0.95f, 0.85f, 0.2f);
	ct.AddPoint(170, 0.9f, 0.3f, 0.9f);
	ct.AddPoint(186, 0.2f, 0.9f, 0.9f);
	ct.AddPoint(202, 0.95f, 0.5f, 0.2f);
	ct.AddPoint(218, 0.4f, 0.95f, 0.5f);
	ct.AddPoint(234, 0.5f, 0.4f, 1.0f);
	ct.AddPoint(255, 1.0f, 0.8f, 0.85f);
	ct.MakeColorTable(colorTableR, colorTableG, colorTableB);
}

int inline GetBlockId(glm::vec3 p) {//(개선+); 시프트 연산자로 블록 아이디 계산
	int x = p.x, y = p.y, z = p.z;
	int bx = x >> BSHIFT, by = y >> BSHIFT, bz = z >> BSHIFT;
	return (bx << 10) | (by << 5) | bz; // 수정A-4: 시프트 복호화로(어차피 진수표현만 상이)
}

// 수정: void -> float. 한 번 도는 데 걸린 ms 를 돌려준다.
// 왜: 바깥에서 여러 번 돌려 평균과 최소를 내려면 값이 필요하다.
float BuildPreIntegrationTable() {
	auto t_start = std::chrono::high_resolution_clock::now();//
	for (int sf = 0; sf < TABLE_SIZE; sf++) {          // front
		for (int sb = 0; sb < TABLE_SIZE; sb++) {      // back
			int idx = sf * TABLE_SIZE + sb;

			float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f, a_sum = 0.0f;

			int n = (sb > sf) ? (sb - sf) : (sf - sb);  // 부분샘플 개수
			if (n == 0) {//밀도 변화 없음
				float a = alphaTable[sf];
				a = 1.0f - powf(1.0f - a, DEFAULT_SEGMENT_LENGTH);
				a_sum = a;
				r_sum = colorTableR[sf] * a;
				g_sum = colorTableG[sf] * a;
				b_sum = colorTableB[sf] * a;
			}
			else {
				float width = (float)n;
				float inv_width = DEFAULT_SEGMENT_LENGTH / width;             // 부분샘플 두께
				int dir = (sb > sf) ? 1 : -1;               // 광선 진행 방향(sf -> sb)

				for (int i = sf; i != sb; i += dir) {
					float alpha = alphaTable[i];
					if (alpha > 0.0f) {
						// alpha correction
						float ca = 1.0f - powf(1.0f - alpha, inv_width);

						r_sum += (1.0f - a_sum) * colorTableR[i] * ca;
						g_sum += (1.0f - a_sum) * colorTableG[i] * ca;
						b_sum += (1.0f - a_sum) * colorTableB[i] * ca;
						a_sum += (1.0f - a_sum) * ca;

						if (a_sum > 0.99f) break;
					}
				}
			}

			// 한 구간을 통째로 합성한 결과를 저장
			preAlpha[idx] = a_sum;
			preColorR[idx] = r_sum;
			preColorG[idx] = g_sum;
			preColorB[idx] = b_sum;
		}
	}
	auto t_end = std::chrono::high_resolution_clock::now();     // 추가
	auto dur = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
	return dur.count() * 0.001f;
}

// 추가: CPU 생성을 여러 번 돌려 평균/최소를 낸다.
// 왜 최소도 보나: 최소값이 "방해받지 않았을 때의 진짜 실력"에 가장 가깝다.
//               평균은 백그라운드 작업에 오염되기 쉽다.
void BenchTableCPU() {
	float sum = 0.0f, lo = 1e9f;
	for (int i = 0; i < TABLE_BENCH_RUNS; i++) {
		float ms = BuildPreIntegrationTable();
		sum += ms;
		if (ms < lo) lo = ms;
		printf("[CPU table] run %d : %.3f ms\n", i, ms);
	}
	printf("[CPU table] mean %.3f ms, min %.3f ms (%dx%d = %d entries, seg=%.2f)\n",
		sum / TABLE_BENCH_RUNS, lo, TABLE_SIZE, TABLE_SIZE,
		TABLE_SIZE * TABLE_SIZE, DEFAULT_SEGMENT_LENGTH);
}

// 추가: CPU판과 GPU판이 같은 표를 만들었는지 숫자로 대조한다.
// 왜: 시간만 재고 값을 안 맞춰보면, 빨리 틀린 답을 낸 것을 자랑하게 된다.
void CompareTables() {
	float maxA = 0.0f, maxR = 0.0f;
	int worst = 0;
	for (int i = 0; i < TABLE_SIZE * TABLE_SIZE; i++) {
		float da = preAlpha[i] - gpuAlpha[i];
		if (da < 0.0f) da = -da;
		float dr = preColorR[i] - gpuColorR[i];
		if (dr < 0.0f) dr = -dr;
		if (da > maxA) { maxA = da; worst = i; }
		if (dr > maxR) maxR = dr;
	}
	printf("[compare] max |alpha diff| = %.9f at idx %d (sf=%d, sb=%d)\n",
		maxA, worst, worst >> 8, worst & 0xFF);
	printf("[compare] max |R diff|     = %.9f\n", maxR);
	printf("[compare] sample idx 30000 : cpu a=%.9f  gpu a=%.9f\n",
		preAlpha[30000], gpuAlpha[30000]);
}

void Render(glm::vec3 eye) {
	using namespace glm;

	glm::vec3 at(128, 128, 112);
	glm::vec3 up(0, 1, 0);

	glm::vec3 w = glm::normalize(at - eye);
	glm::vec3 u = glm::normalize(glm::cross(up, w));
	glm::vec3 v = glm::normalize(glm::cross(w, u));


	auto start = std::chrono::high_resolution_clock::now();
	const float supersampling = 0.5f * (512.0f / WIDTH);
	/////////////////레이캐스팅
	for (int y = 0; y < HEIGHT; y++) { // 영상의 y좌표
		for (int x = 0; x < WIDTH; x++) { // 영상의 x좌표
			glm::vec3 RS = eye + u * (x - WIDTH * 0.5f) * supersampling + v * (y - HEIGHT * 0.5f) * supersampling;

			float t1, t2; // 한 구간
			t1 = -RS.x / w.x;
			t2 = (255 - RS.x) / w.x;
			float xm = __min(t1, t2);
			float xM = __max(t1, t2);

			t1 = -RS.y / w.y;
			t2 = (255 - RS.y) / w.y;
			float ym = __min(t1, t2);
			float yM = __max(t1, t2);

			t1 = -RS.z / w.z;
			t2 = (224 - RS.z) / w.z;
			float zm = __min(t1, t2);
			float zM = __max(t1, t2);
			float tm = __max(__max(xm, ym), zm);
			float tM = __min(__min(xM, yM), zM);

			float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f;
			float a_sum = 0.0f;

			// 수정: 하드코딩 0.5 -> 공통 상수. CPU 경로와 GPU 경로가 갈라지지 않게.
			const float step = DEFAULT_SEGMENT_LENGTH;
			for (float t = tm; t < tM; t = t + step) { // 광선을 진행하자
				glm::vec3 p = RS + w * t;
				if (isOutside(p))
					continue;

				// 내(p)가 속한 블록의 min, max 안다고 가정.
				int bid = GetBlockId(p); // 123456
				// 수정A-5: 비트연산자 활용해 봄.
				int bz = bid & 0x1F;
				int by = (bid >> 5) & 0x1F;
				int bx = (bid >> 10) & 0x1F;
				int min_value = bm[bz][by][bx];
				int max_value = bM[bz][by][bx];
				if (sumTable[max_value] - sumTable[min_value - 1] == 0) {
					float jump = 0;
					int nextBid;
					// 투명한 블록이니까, 연산을 건너뛰자. 광선을 빠르게 전진하자.
					do {
						jump += 1.0f;
						nextBid = GetBlockId(p + w * jump); // 추가 전진
					} while (bid == nextBid);
					t = t + (jump - step);
					continue;
				}


				int d = GetDensity(p);

				float alpha = alphaTable[d];
				if (alpha == 0)
					continue;
				alpha = 1 - pow((1 - alpha), step); // alpha-correction


				float r = colorTableR[d];
				float g = colorTableG[d];
				float b = colorTableB[d];

				//조명; 중앙차분법
				float dx = (GetDensity(p + vec3(1, 0, 0)) - GetDensity(p - vec3(1, 0, 0))) * 0.5f;
				float dy = (GetDensity(p + vec3(0, 1, 0)) - GetDensity(p - vec3(0, 1, 0))) * 0.5f;
				float dz = (GetDensity(p + vec3(0, 0, 1)) - GetDensity(p - vec3(0, 0, 1))) * 0.5f;

				vec3 V = w;
				vec3 N(dx, dy, dz), L = w;

				if (length(N) > 0.0f) N = normalize(N);
				vec3 H = normalize(L + V);
				float NL = fabs(dot(N, L));
				float NH = fabs(dot(N, H));

				float Ia = 0.25f, Id = 0.5f, Is = 0.9f;

				glm::vec3 Ka(r * 0.8f, g * 0.8f, b * 0.8f);
				glm::vec3 Kd(r, g, b);
				glm::vec3 Ks(1.2f, 0.8f, 0.8f); //오팔 느낌

				glm::vec3 I = Ia * Ka + Id * Kd * NL + Is * Ks * pow(NH, 30.0f);
				I = glm::clamp(I, 0.0f, 1.0f);//__min

				r_sum += (1.0f - a_sum) * (I.r * alpha);
				g_sum += (1.0f - a_sum) * (I.g * alpha);
				b_sum += (1.0f - a_sum) * (I.b * alpha);
				a_sum += (1.0f - a_sum) * alpha;
				if (a_sum > 0.99f) break; // 조기 광선 종료, early ray termination

			}
			MyTexture[y][x][0] = int(r_sum * 255);
			MyTexture[y][x][1] = int(g_sum * 255);
			MyTexture[y][x][2] = int(b_sum * 255);

		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "실행 시간: " << duration.count() * 0.001f << " ms" << std::endl;
}

extern "C" int cuInit();
extern "C" float cuBuildPreIntegrationTable();   // 추가

void MyInit() {
	glClearColor(0.0, 0.0, 0.0, 0.0);
	FileRead();
	GenBlocks(); // 파일은 읽고 난 다음에.
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
	glEnable(GL_TEXTURE_2D);

	InitTables();

	BenchTableCPU();   // 수정: BuildPreIntegrationTable() 1회 -> 반복 측정
	cuInit();          // 여기서 alphaTable, colorTable 이 GPU 로 올라간다

	cuBuildPreIntegrationTable();  // 추가: 같은 표를 GPU 로 한 번 더 만든다
	CompareTables();               // 추가: 두 표가 같은지 숫자로 확인
}

extern "C" int cumain(float ex, float ey, float ez);
void MyDisplay() {
	////////////////카메라 세팅
	static float t = 0;
	t += 1.0;

#if FIX_CAMERA
	glm::vec3 eye(0, 0, 100);   // 측정용: 한 자리에 고정
#else
	glm::vec3 eye(sin(t * 0.1) * 50, 0, 100);
#endif
	//cout << glm::to_string(eye) << endl;   // 측정 중엔 콘솔 출력이 방해가 된다

	//Render(eye);
	cumain(eye.x, eye.y, eye.z);
	glTexImage2D(GL_TEXTURE_2D, 0, 3, WIDTH, HEIGHT, 0, GL_RGB,
		GL_UNSIGNED_BYTE, &MyTexture[0][0][0]);

	glClear(GL_COLOR_BUFFER_BIT);
	glBegin(GL_QUADS);
	float fSize = 0.8f;
	glTexCoord2f(0.0, 0.0); glVertex3f(-fSize, -fSize, 0.0);
	glTexCoord2f(0.0, 1.0); glVertex3f(-fSize, fSize, 0.0);
	glTexCoord2f(1.0, 1.0); glVertex3f(fSize, fSize, 0.0);
	glTexCoord2f(1.0, 0.0); glVertex3f(fSize, -fSize, 0.0);
	glEnd();
	glutSwapBuffers();
}

int main(int argc, char** argv) {
	glutInit(&argc, argv); //GLUT 윈도우 함수
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutCreateWindow("OpenGL Drawing Example");
	MyInit();
	glutDisplayFunc(MyDisplay);
	glutIdleFunc(MyDisplay);
	glutMainLoop();

	return 0;
}