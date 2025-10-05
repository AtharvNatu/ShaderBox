#include "Ocean.hpp"

//* Get K Vector From Mesh Grid (n, m)
#define K_VEC(n, m) vmath::vec2(2 * M_PI * (n - N / 2) / x_length, 2 * M_PI * (m - M / 2) / z_length)

Ocean::Ocean(int _N, int _M, float _x_length, float _z_length, vmath::vec2 _omega, float _V, float _A, float _lambda)
{
    N = _N;
    M = _M;
    omega_hat = vmath::normalize(_omega);
    V = _V;
    A = _A;
    x_length = _x_length;
    z_length = _z_length;
    lambda = _lambda;

    generator.seed(time(nullptr));
    kNum = N * M;

    displacement_map = new vmath::vec3[kNum];
    normal_map = new vmath::vec3[kNum];

    h_twiddle_0 = new std::complex<float>[kNum];
    h_twiddle_0_conjunction = new std::complex<float>[kNum];
    h_twiddle = new std::complex<float>[kNum];

    //! Initialize h_twiddle_0 and h_twiddle_0_conjunction in Eqn. 26
    for (int n = 0; n < N; n++)
    {
        for (int m = 0; m < M; m++)
        {
            int index = m * N + n;
            vmath::vec2 k = K_VEC(n, m);
            h_twiddle_0[index] = func_h_twiddle_0(k);
            // h_twiddle_0_conjunction[index] = std::conj(func_h_twiddle_0(k));

            int kn_neg = (N - n) % N;
            int km_neg = (M - m) % M;
            int index_neg = km_neg * N + kn_neg;

            h_twiddle_0_conjunction[index] = std::conj(h_twiddle_0[index_neg]);
        }
    }
}

//! Eqn. 14
inline float Ocean::omega(float k) const
{
    return sqrt(G * k);
}

//! Eqn. 23 Phillips Spectrum
inline float Ocean::phillips_spectrum(vmath::vec2 k) const
{
    // Code
    float k_length = vmath::length(k);
    if (k_length < 1e-6f)
        return 0.0f;

    // Largest possible waves from continuous wind of speed V
    float wave_length = (V * V) / G;
    vmath::vec2 k_hat = vmath::normalize(k);

    float dot_k_hat_omega_hat = vmath::dot(k_hat, omega_hat);
    float dot_term = dot_k_hat_omega_hat * dot_k_hat_omega_hat;

    float exp_term = expf(-1.0f / (k_length * k_length * wave_length * wave_length));
    float result = A * exp_term * dot_term / powf(k_length, 4.0f);

    // Small-wave damping (Eq. 24) — uses constant L
    result *= expf(-k_length * k_length * L * L);

    return result;
}

//! Eqn. 25
inline std::complex<float> Ocean::func_h_twiddle_0(vmath::vec2 k)
{
    // Code
    float xi_r = normal_distribution(generator);
    float xi_i = normal_distribution(generator);

    return sqrt(0.5f) * std::complex<float>(xi_r, xi_i) * sqrt(phillips_spectrum(k));
}

//! Eqn. 26
inline std::complex<float> Ocean::func_h_twiddle(int kn, int km, float t) const
{
    // Code
    int index = km * N + kn;

    float k = vmath::length(K_VEC(kn, km));

    std::complex<float> term1 = h_twiddle_0[index] * exp(std::complex<float>(0.0f, omega(k) * t));
    std::complex<float> term2 = h_twiddle_0_conjunction[index] * exp(std::complex<float>(0.0f, -omega(k) * t));

    return term1 + term2;
}

//! Eqn. 19
void Ocean::generate_fft_data(float time)
{
    // Variable Declarations
    fftwf_complex *in_height = nullptr;
    fftwf_complex *in_slope_x = nullptr, *in_slope_z = nullptr;
    fftwf_complex *in_displacement_x = nullptr, *in_displacement_z = nullptr;

    fftwf_complex *out_height = nullptr;
    fftwf_complex *out_slope_x = nullptr, *out_slope_z = nullptr;
    fftwf_complex *out_displacement_x = nullptr, *out_displacement_z = nullptr;

    fftwf_plan p_height, p_slope_x, p_slope_z, p_displacement_x, p_displacement_z;

    // Code

    //! Eqn. 20 ikh_twiddle
    std::complex<float>* slope_x_term = new std::complex<float>[kNum];
    std::complex<float>* slope_z_term = new std::complex<float>[kNum];

    //! Eqn. 29
    std::complex<float>* displacement_x_term = new std::complex<float>[kNum];
    std::complex<float>* displacement_z_term = new std::complex<float>[kNum];

//     for (int n = 0; n < 4; ++n)
//   for (int m = 0; m < 4; ++m) {
//     auto v = func_h_twiddle(n, m, 0.0f);
//     fprintf(gpFile, "[%d,%d] %f %f\n", n, m, v.real(), v.imag());
//   }

    for (int n = 0; n < N; n++)
    {
        for (int m = 0; m < M; m++)
        {
            int index = m * N + n;

            h_twiddle[index] = func_h_twiddle(n, m, time);

            vmath::vec2 k_vec = K_VEC(n, m);
            float k_length = vmath::length(k_vec);
            vmath::vec2 k_vec_normalized = k_length == 0 ? k_vec : vmath::normalize(k_vec);

            slope_x_term[index] = std::complex<float>(0, k_vec[0]) * h_twiddle[index];
            slope_z_term[index] = std::complex<float>(0, k_vec[1]) * h_twiddle[index];
            
            displacement_x_term[index] = std::complex<float>(0, -k_vec_normalized[0]) * h_twiddle[index];
            displacement_z_term[index] = std::complex<float>(0, -k_vec_normalized[1]) * h_twiddle[index];
        }
    }

    //* Prepare FFT Input and Output
    in_height = (fftwf_complex*)h_twiddle;
    in_slope_x = (fftwf_complex*)slope_x_term;
    in_slope_z = (fftwf_complex*)slope_z_term;
    in_displacement_x = (fftwf_complex*)displacement_x_term;
    in_displacement_z = (fftwf_complex*)displacement_z_term;

    out_height = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * kNum);
    out_slope_x = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * kNum);
    out_slope_z = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * kNum);
    out_displacement_x = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * kNum);
    out_displacement_z = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * kNum);
    
    p_height = fftwf_plan_dft_2d(
        N, 
        M, 
        in_height, 
        out_height, 
        FFTW_BACKWARD, 
        FFTW_ESTIMATE
    );

    p_slope_x = fftwf_plan_dft_2d(
        N,
        M,
        in_slope_x,
        out_slope_x,
        FFTW_BACKWARD,
        FFTW_ESTIMATE
    );

    p_slope_z = fftwf_plan_dft_2d(
        N,
        M,
        in_slope_z,
        out_slope_z,
        FFTW_BACKWARD,
        FFTW_ESTIMATE
    );

    p_displacement_x = fftwf_plan_dft_2d(
        N,
        M,
        in_displacement_x,
        out_displacement_x,
        FFTW_BACKWARD,
        FFTW_ESTIMATE
    );

    p_displacement_z = fftwf_plan_dft_2d(
        N,
        M,
        in_displacement_z,
        out_displacement_z,
        FFTW_BACKWARD,
        FFTW_ESTIMATE
    );

    fftwf_execute(p_height);
    fftwf_execute(p_slope_x);
    fftwf_execute(p_slope_z);
    fftwf_execute(p_displacement_x);
    fftwf_execute(p_displacement_z);

    for (int n = 0; n < N; n++)
    {
        for (int m = 0; m < M; m++)
        {
            int index = m * N + n;
            float sign = 1;

            // Flip the sign
            if ((m + n) % 2)
                sign = -1;

            normal_map[index] = vmath::normalize(vmath::vec3(
                sign * out_slope_x[index][0],
                -1,
                sign * out_slope_z[index][0]
            ));

            displacement_map[index] = vmath::vec3(
                (n - N / 2) * x_length / N - sign * lambda * out_displacement_x[index][0],
                sign * out_height[index][0],
                (m - M / 2) * z_length / M - sign * lambda * out_displacement_z[index][0]
            );
        }
    }

    fftwf_destroy_plan(p_displacement_z);
    fftwf_destroy_plan(p_displacement_x);
    fftwf_destroy_plan(p_slope_z);
    fftwf_destroy_plan(p_slope_x);
    fftwf_destroy_plan(p_height);

    fftwf_free(out_displacement_z);
    out_displacement_z = nullptr;

    fftwf_free(out_displacement_x);
    out_displacement_x = nullptr;

    fftwf_free(out_slope_z);
    out_slope_z = nullptr;

    fftwf_free(out_slope_x);
    out_slope_x = nullptr;

    fftwf_free(out_height);
    out_height = nullptr;

    delete[] displacement_z_term;
    displacement_z_term = nullptr;

    delete[] displacement_x_term;
    displacement_x_term = nullptr;

    delete[] slope_z_term;
    slope_z_term = nullptr;

    delete[] slope_x_term;
    slope_x_term = nullptr;

}

Ocean::~Ocean()
{
    if (h_twiddle)
    {
        delete[] h_twiddle;
        h_twiddle = nullptr;
    }

    if (h_twiddle_0_conjunction)
    {
        delete[] h_twiddle_0_conjunction;
        h_twiddle_0_conjunction = nullptr;
    }

    if (h_twiddle_0)
    {
        delete[] h_twiddle_0;
        h_twiddle_0 = nullptr;
    }

    if (normal_map)
    {
        delete[] normal_map;
        normal_map = nullptr;
    }

    if (displacement_map)
    {
        delete[] displacement_map;
        displacement_map = nullptr;
    }
}