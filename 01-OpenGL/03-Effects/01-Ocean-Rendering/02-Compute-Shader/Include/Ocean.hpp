#ifndef OCEAN_HPP
#define OCEAN_HPP

#include <random>
#include <complex>

#define _USE_MATH_DEFINES   1
#include <cmath>

#include "Common.hpp"
#include "fftw3.h"

class Ocean
{
    private:
        std::complex<float>* h_twiddle_0 = nullptr;
        std::complex<float>* h_twiddle_0_conjunction = nullptr;
        std::complex<float>* h_twiddle = nullptr;

        std::default_random_engine generator;
        std::normal_distribution<float> normal_distribution{0.0f, 1.0f};

        const float PI = float(M_PI);
        const float G = 9.8f;   // Gravitational Constant
        const float L = 0.1;

        float A, V, lambda;
        float x_length, z_length;
        int N, M, kNum;
        vmath::vec2 omega_hat;

        inline float omega(float k) const;
        inline float phillips_spectrum(vmath::vec2 k) const;
        inline std::complex<float> func_h_twiddle_0(vmath::vec2 k);
        inline std::complex<float> func_h_twiddle(int kn, int km, float t) const;

    public:
        // N, M                 ->  Resolution
        // x_length, z_length   ->  Actual grid lengths (meters)
        // omega_hat            ->  Direction of wind
        // V                    ->  Speed of wind
        Ocean(int N, int M, float x_length, float z_length, vmath::vec2 omega, float V, float A, float lambda);
        ~Ocean();

        void generate_fft_data(float time);

        vmath::vec3* displacement_map;
        vmath::vec3* normal_map;

};


#endif  // OCEAN_HPP
