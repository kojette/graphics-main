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

unsigned char MyTexture[HEIGHT][WIDTH][3];
unsigned char vol[VOLZ][VOLY][VOLX];

float alphaTable[256];
float colorTableR[256];
float colorTableG[256];
float colorTableB[256];

//pre-integration
#define DEFAULT_SEGMENT_LENGTH 0.5f   // 기존 코드의 step=0.5 와 동일
#define TABLE_SIZE 256         
// 알파, 컬러의 적분
float preT[TABLE_SIZE];
float preKr[TABLE_SIZE], preKg[TABLE_SIZE], preKb[TABLE_SIZE];
// 좌표축이 (sf, sb) 두 개인 2D 테이블. CPU에서 만든 뒤 GPU로 1회 업로드.
// 배열은 1차원으로
float preAlpha[TABLE_SIZE * TABLE_SIZE];
float preColorR[TABLE_SIZE * TABLE_SIZE];
float preColorG[TABLE_SIZE * TABLE_SIZE];
float preColorB[TABLE_SIZE * TABLE_SIZE];

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


void InitTables() {
	AlphaTable  mat;
	mat.AddPoint(0, 0.0f);//공기&연조직 비가시
	mat.AddPoint(90, 0.0f);
	mat.AddPoint(120, 0.2f);//뼈? 
	mat.AddPoint(150, 0.6f);
	mat.AddPoint(200, 1.0f);//확실히 뼈
	mat.AddPoint(255, 1.0f);
	mat.MakeAlphaTable(alphaTable);

	ColorTable ct;
	ct.AddPoint(0, 0.0f, 0.0f, 0.0f);
	ct.AddPoint(120, 0.1f, 0.2f, 0.55f);   // 딥블루 오팔베이스
	ct.AddPoint(160, 0.85f, 0.5f, 0.2f);   // 블루 시안
	ct.AddPoint(200, 0.7f, 0.75f, 0.3f);
	ct.AddPoint(220, 0.97f, 0.7f, 0.8f);
	ct.AddPoint(255, 1.0f, 0.8f, 0.85f);  // 핑크
	ct.MakeColorTable(colorTableR, colorTableG, colorTableB);
}

void BuildPreIntegrationTable(float segmentLength) {
	//누적 적분
	preT[0] = 0.0f;
	preKr[0] = preKg[0] = preKb[0] = 0.0f;
	for (int s = 1; s < TABLE_SIZE; s++) {
		// preT:alphaTable
		preT[s] = preT[s - 1] + 0.5f * (alphaTable[s - 1] + alphaTable[s]);
		// preK:associated color의 누적
		float ar0 = colorTableR[s - 1] * alphaTable[s - 1];
		float ar1 = colorTableR[s] * alphaTable[s];
		float ag0 = colorTableG[s - 1] * alphaTable[s - 1];
		float ag1 = colorTableG[s] * alphaTable[s];
		float ab0 = colorTableB[s - 1] * alphaTable[s - 1];
		float ab1 = colorTableB[s] * alphaTable[s];
		preKr[s] = preKr[s - 1] + 0.5f * (ar0 + ar1);
		preKg[s] = preKg[s - 1] + 0.5f * (ag0 + ag1);
		preKb[s] = preKb[s - 1] + 0.5f * (ab0 + ab1);
	}

	//2차원 테이블
	for (int sf = 0; sf < TABLE_SIZE; sf++) {
		for (int sb = 0; sb < TABLE_SIZE; sb++) {
			int idx = sf * TABLE_SIZE + sb;
			float tau_avg, cr_avg, cg_avg, cb_avg;

			if (sf == sb) {//알파값은 대칭 최적화 해도 될듯?
				tau_avg = alphaTable[sf];
				cr_avg = colorTableR[sf] * alphaTable[sf];
				cg_avg = colorTableG[sf] * alphaTable[sf];
				cb_avg = colorTableB[sf] * alphaTable[sf];
			}
			else {
				float inv_diff = 1.0f / (float)(sb - sf);
				tau_avg = (preT[sb] - preT[sf]) * inv_diff;
				cr_avg = (preKr[sb] - preKr[sf]) * inv_diff;
				cg_avg = (preKg[sb] - preKg[sf]) * inv_diff;
				cb_avg = (preKb[sb] - preKb[sf]) * inv_diff;
			}

			float alphaVal = 1.0f - expf(-segmentLength * tau_avg);
			preAlpha[idx] = alphaVal;
			// 흡수 무시 근사: C ≈ c_avg * d
			preColorR[idx] = cr_avg * segmentLength;
			preColorG[idx] = cg_avg * segmentLength;
			preColorB[idx] = cb_avg * segmentLength;
		}
	}
}

extern "C" int cuInit();
void MyInit() {
	glClearColor(0.0, 0.0, 0.0, 0.0);
	FileRead();
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
	glEnable(GL_TEXTURE_2D);

	InitTables();
	BuildPreIntegrationTable(DEFAULT_SEGMENT_LENGTH);
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