#include <SDL.h>
#include <SDL_image.h>
#include <SDL_ttf.h>
#include <SDL_mixer.h>
#include <iostream>
#include <stdlib.h>  
#include <crtdbg.h>   //for malloc and free
#include <set>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <stdio.h>
#include <vector>
#include <string>
#define _CRTDBG_MAP_ALLOC
#ifdef _DEBUG
#define new new( _NORMAL_BLOCK, __FILE__, __LINE__)
#endif

const int WIDTH = 800;
const int HEIGHT = 600;
SDL_Window* window;
SDL_Renderer* renderer;
bool running;
SDL_Event event;
std::set<std::string> keys;
std::set<std::string> currentKeys;
int mouseX = 0;
int mouseY = 0;
int mouseDeltaX = 0;
int mouseDeltaY = 0;
int mouseScroll = 0;
std::set<int> buttons;
std::set<int> currentButtons;
Uint64 counter = 0;

__device__ struct Pair {
	int16_t x, y;
};
__device__ Pair pairs[4] = { {1, 0}, {0, 1}, {-1, 0}, {0, -1} };

class Pixel {
public:
	uint8_t life = 0;
	uint8_t r = 0, g = 0, b = 0;
	uint16_t x = 0, y = 0;
	void setPos(uint16_t x, uint16_t y) {
		this->x = x;
		this->y = y;
	}
	__device__ void setColor(uint8_t r, uint8_t g, uint8_t b) {
		this->r = r;
		this->g = g;
		this->b = b;
	}
	void draw(Uint32* pixel_ptr, SDL_PixelFormat* format) {
		if (life > 0) {
			pixel_ptr[y * WIDTH + x] = SDL_MapRGB(format, r, g, b);
			life--;
			if (life == 0) {
				r = 0;
				g = 0;
				b = 0;
			}
		}
		else if (r != 0 || g != 0 || b != 0) {
			life = 3;
		}
	}
	__device__ Pair spread(curandState* state) {
		Pair dir = pairs[curand(state) % 4];

		if (x == 0 && dir.x == -1) {
			dir.x = WIDTH - 1;
		}
		else if (x == WIDTH - 1 && dir.x == 1) {
			dir.x = 0;
		}
		else {
			dir.x += x;
		}

		if (y == 0 && dir.y == -1) {
			dir.y = HEIGHT - 1;
		}
		else if (y == HEIGHT - 1 && dir.y == 1) {
			dir.y = 0;
		}
		else {
			dir.y += y;
		}

		return dir;
	}
};
Pixel pixels[HEIGHT * WIDTH];
Pixel* d_pixels;
size_t p_size = sizeof(Pixel) * size_t(HEIGHT * WIDTH);

SDL_Surface* infoSurface, * saveSurface;
unsigned char* savePixels;


void debug(int line, std::string file) {
	std::cout << "Line " << line << " in file " << file << ": " << SDL_GetError() << std::endl;
}

__global__ void initCurand(unsigned int seed, curandState* state) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init(seed, idx, 0, &state[idx]);
}

__global__ void spread(Pixel pixels[HEIGHT * WIDTH], curandState* state) {
	Pixel* thisPixel = &pixels[threadIdx.x * WIDTH + blockIdx.x];
	if (thisPixel->life > 0) {
		Pair direction = thisPixel->spread(state);
		Pixel* toSpread = &pixels[direction.y * WIDTH + direction.x];
		if (toSpread->life == 0) {
			toSpread->setColor(thisPixel->r + (curand(state) % 3 - 1), thisPixel->g + (curand(state) % 3 - 1), thisPixel->b + (curand(state) % 3 - 1));
		}
	}
}
Uint32 frameStart, spreadStart, drawStart;
int frameTime = 0;
bool timing = true;

int main(int argc, char* argv[]) {
	srand(time(0));
	if (SDL_Init(SDL_INIT_EVERYTHING) == 0 && TTF_Init() == 0 && Mix_OpenAudio(44100, MIX_DEFAULT_FORMAT, 2, 2048) == 0) {
		//Setup
		window = SDL_CreateWindow("Window", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, 0);
		if (window == NULL) {
			debug(__LINE__, __FILE__);
		}

		renderer = SDL_CreateRenderer(window, -1, 0);
		if (renderer == NULL) {
			debug(__LINE__, __FILE__);
		}

		infoSurface = SDL_GetWindowSurface(window);
		SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_NONE);
		savePixels = new unsigned char[infoSurface->w * infoSurface->h * infoSurface->format->BytesPerPixel];

		SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888,
			SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);
		void* txtPixels;
		int pitch;
		SDL_PixelFormat* format = SDL_AllocFormat(SDL_PIXELFORMAT_RGBA8888);
		Uint32* pixel_ptr;

		for (uint16_t i = 0; i < WIDTH; i++) {
			for (uint16_t j = 0; j < HEIGHT; j++) {
				pixels[j * WIDTH + i].setPos(i, j);
			}
		}

		Pixel* first = &pixels[(HEIGHT + 1) * WIDTH / 2];
		first->r = 127;
		first->g = 127;
		first->b = 127;

		cudaSetDevice(0);
		curandState* d_state;
		cudaMalloc(&d_state, sizeof(curandState));
		initCurand << <1, 1 >> > (time(0), d_state);
		cudaDeviceSynchronize();
		cudaMalloc((void**)&d_pixels, p_size);

		//Main loop
		running = true;
		while (running) {
			frameStart = SDL_GetTicks();
			//handle events
			for (std::string i : keys) {
				currentKeys.erase(i); //make sure only newly pressed keys are in currentKeys
			}
			for (int i : buttons) {
				currentButtons.erase(i); //make sure only newly pressed buttons are in currentButtons
			}
			mouseScroll = 0;
			while (SDL_PollEvent(&event)) {
				switch (event.type) {
				case SDL_QUIT:
					running = false;
					break;
				case SDL_KEYDOWN:
					if (!keys.contains(std::string(SDL_GetKeyName(event.key.keysym.sym)))) {
						currentKeys.insert(std::string(SDL_GetKeyName(event.key.keysym.sym)));
					}
					keys.insert(std::string(SDL_GetKeyName(event.key.keysym.sym))); //add keydown to keys set
					break;
				case SDL_KEYUP:
					keys.erase(std::string(SDL_GetKeyName(event.key.keysym.sym))); //remove keyup from keys set
					break;
				case SDL_MOUSEMOTION:
					mouseX = event.motion.x;
					mouseY = event.motion.y;
					mouseDeltaX = event.motion.xrel;
					mouseDeltaY = event.motion.yrel;
					break;
				case SDL_MOUSEBUTTONDOWN:
					if (!buttons.contains(event.button.button)) {
						currentButtons.insert(event.button.button);
					}
					buttons.insert(event.button.button);
					break;
				case SDL_MOUSEBUTTONUP:
					buttons.erase(event.button.button);
					break;
				case SDL_MOUSEWHEEL:
					mouseScroll = event.wheel.y;
					break;
				}
			}

			spreadStart = SDL_GetTicks();
			cudaMemcpy(d_pixels, pixels, p_size, cudaMemcpyHostToDevice);
			spread << <WIDTH, HEIGHT >> > (d_pixels, d_state);
			cudaDeviceSynchronize();
			cudaMemcpy(pixels, d_pixels, p_size, cudaMemcpyDeviceToHost);
			if (timing) {
				std::cout << "spread time: " << SDL_GetTicks() - spreadStart;
			}

			drawStart = SDL_GetTicks();
			SDL_LockTexture(texture, NULL, &txtPixels, &pitch);
			pixel_ptr = (Uint32*)txtPixels;
			for (uint16_t i = 0; i < WIDTH; i++) {
				for (uint16_t j = 0; j < HEIGHT; j++) {
					pixels[j* WIDTH + i].draw(pixel_ptr, format);
				}
			}
			SDL_UnlockTexture(texture);
			SDL_RenderCopy(renderer, texture, NULL, NULL);
			SDL_RenderReadPixels(renderer, &infoSurface->clip_rect, infoSurface->format->format, savePixels, infoSurface->w * infoSurface->format->BytesPerPixel);
			SDL_RenderPresent(renderer);
			if (timing) {
				std::cout << " draw time: " << SDL_GetTicks() - drawStart;
			}


			saveSurface = SDL_CreateRGBSurfaceFrom(savePixels, infoSurface->w, infoSurface->h, infoSurface->format->BitsPerPixel, infoSurface->w * infoSurface->format->BytesPerPixel,
				infoSurface->format->Rmask, infoSurface->format->Gmask, infoSurface->format->Bmask, infoSurface->format->Amask);
			//SDL_SaveBMP_RW(saveSurface, SDL_RWFromFile(("Images/Image" + std::to_string(counter) + ".bmp").c_str(), "wb"), 1);
			SDL_FreeSurface(saveSurface);
			counter++;

			frameTime = SDL_GetTicks() - frameStart;
			std::cout << " frame time: " << frameTime << std::endl;
			//cudaError_t err = cudaGetLastError();
			//std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
		}
		//Clean up
		delete[] savePixels;
		SDL_FreeFormat(format);
		cudaFree(d_pixels);
		cudaFree(d_state);
		if (window) {
			SDL_DestroyWindow(window);
		}
		if (renderer) {
			SDL_DestroyRenderer(renderer);
		}
		TTF_Quit();
		Mix_Quit();
		IMG_Quit();
		SDL_Quit();
		return 0;
	}
	else {
		return 0;
	}
}