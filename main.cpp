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
#define WIDTH   1024
#define HEIGHT  1024
#define VOLX 256
#define VOLY 256	
#define VOLZ 225
const int BSIZE = 8;
const int BSHIFT = 3;

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
			//가시성을 위한? 것 같은 교수님st?
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
				for (int z = bz * BSIZE; z <= __min(bz * BSIZE + BSIZE, VOLZ - 1); z++) { // 28*8 = for 224      z<232      vol[226]
					for (int y = by * BSIZE; y <= __min(by * BSIZE + BSIZE, VOLY - 1); y++) {
						for (int x = bx * BSIZE; x <= __min(bx * BSIZE + BSIZE, VOLX - 1); x++) { //bx=31, 31*8=248~256
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
	// y,z
	// linear interpolation : 직선형 보간
	// cubic interpolation : 3차 함수를 이용한 보간
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
	AlphaTable  mat;
	mat.AddPoint(0, 0.0f);//공기&연조직 비가시
	mat.AddPoint(90, 0.0f);
	mat.AddPoint(120, 0.2f);//뼈? 
	mat.AddPoint(150, 0.6f);
	mat.AddPoint(200, 1.0f);//확실히 뼈
	mat.AddPoint(255, 1.0f);
	mat.MakeAlphaTable(alphaTable);

	sumTable[0] = alphaTable[0];
	for (int i = 1; i < 256; i++) {
		sumTable[i] = sumTable[i - 1] + alphaTable[i];
	}

	//in : alphaTable
	//out : sumTable


	ColorTable ct;
	ct.AddPoint(0, 0.0f, 0.0f, 0.0f);
	ct.AddPoint(120, 0.1f, 0.2f, 0.55f);   // 딥블루 오팔베이스
	ct.AddPoint(160, 0.85f, 0.5f, 0.2f);   // 블루 시안
	ct.AddPoint(200, 0.7f, 0.75f, 0.3f);
	ct.AddPoint(220, 0.97f, 0.7f, 0.8f);
	ct.AddPoint(255, 1.0f, 0.8f, 0.85f);  // 핑크
	ct.MakeColorTable(colorTableR, colorTableG, colorTableB);
}

int inline GetBlockId(glm::vec3 p) {//(개선+); 시프트 연산자로 블록 아이디 계산
	int x = p.x, y = p.y, z = p.z;
	int bx = x >> BSHIFT, by = y >> BSHIFT, bz = z >> BSHIFT;
	return (bx << 10) | (by << 5) | bz; // 수정A-4: 시프트 복호화로(어차피 진수표현만 상이)
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

			const float step = 0.5;
			for (float t = tm; t < tM; t = t + step) { // 광선을 진행하자
				glm::vec3 p = RS + w * t;
				if (isOutside(p))
					continue;

				// 내(p)가 속한 블록의 min, max 안다고 가정.
				int bid = GetBlockId(p); // 123456
				// 수정A-5: 비트연산자 활용해 봄. 
				int bz = bid & 0x1F; //1F(16+15)임 즉, 11111이고 &연산함. (0x20-0x01해도?)             
				int by = (bid >> 5) & 0x1F; // 5개 지우고 남은 오른쪽 5개 추출
				int bx = (bid >> 10) & 0x1F; // 이하 동일
				int min_value = bm[bz][by][bx];
				int max_value = bM[bz][by][bx];
				if (sumTable[max_value] - sumTable[min_value - 1] == 0) {
					float jump = 0;
					int nextBid;
					// 투명한 블록임
					//투명한 블록이니까, 연산을 건너뛰자. 광선을 빠르게 전진하자.
					do {
						jump += 1.0f;
						nextBid = GetBlockId(p + w * jump); // 추가 전진
					} while (bid == nextBid);
					t = t + (jump - step);
					continue;
				}


				int d = GetDensity(p); // vol[(int)(p.z)][int(p.y)][int(p.x)]; // d = x선 흡수도, d가 높으면 뼈 d가 낮으면 근육, 지방, 공기

				float alpha = alphaTable[d]; // getAlpha(50, 200, d);
				if (alpha == 0)
					continue;
				alpha = 1 - pow((1 - alpha), step); // alpha-correction


				float r = colorTableR[d]; // d / 255.0;
				float g = colorTableG[d];
				float b = colorTableB[d];

				//조명; 중앙차분법
				float dx = (GetDensity(p + vec3(1, 0, 0)) - GetDensity(p - vec3(1, 0, 0))) * 0.5f; // 수정: 0.5 -> 0.5f (float 리터럴)
				float dy = (GetDensity(p + vec3(0, 1, 0)) - GetDensity(p - vec3(0, 1, 0))) * 0.5f; // 수정: 0.5 -> 0.5f
				float dz = (GetDensity(p + vec3(0, 0, 1)) - GetDensity(p - vec3(0, 0, 1))) * 0.5f; // 수정: 0.5 -> 0.5f

				vec3 V = w;
				vec3 N(dx, dy, dz), L = w;// (1, 0, 0);

				if (length(N) > 0.0f) N = normalize(N);
				vec3 H = normalize(L + V);
				float NL = fabs(dot(N, L));
				float NH = fabs(dot(N, H));



				float Ia = 0.25f, Id = 0.5f, Is = 0.9f;  //살짝 밝게 // 합이 1인게 좋은데 여러 표현 가능


				glm::vec3 Ka(r * 0.8f, g * 0.8f, b * 0.8f); //주변광 반사율 0.8 곱(어두운 배경 연출) // k = 반사율 = 색상(중요!!!!!! 놀랍군)
				glm::vec3 Kd(r, g, b);// = c;
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
void MyInit() {
	glClearColor(0.0, 0.0, 0.0, 0.0);
	FileRead();
	GenBlocks(); // 파일은 읽고 난 다음에.
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
	glEnable(GL_TEXTURE_2D);

	InitTables();
	cuInit();
}
extern "C" int cumain(float ex, float ey, float ez);
void MyDisplay() {
	////////////////카메라 세팅
	static float t = 0;
	t += 1.0;
	glm::vec3 eye(sin(t * 0.1) * 50, 0, 100);
	cout << glm::to_string(eye) << endl;

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
	//cumain();
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