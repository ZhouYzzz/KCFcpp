#ifndef FFTEG_H_
#define FFTEG_H_

#include "fftw3.h"
#include "opencv2/opencv.hpp"

class FFTEngine {
public:
  // FFTEngine(const cv::Mat& src, cv::Mat& dst) {
  //   int H = src.rows; int W = src.cols; int C = src.channels();
  //   int n[2] = {H, W};
  //   p_ = fftwf_plan_many_dft(2, n, C,
  //         (fftwf_complex*)src.data, n, 1, H*W,
  //         (fftwf_complex*)dst.data, n, 1, H*W,
  //         FFTW_FORWARD, FFTW_ESTIMATE);
  //   ip_ = fftwf_plan_many_dft(2, n, C,
  //         (fftwf_complex*)src.data, n, 1, H*W,
  //         (fftwf_complex*)dst.data, n, 1, H*W,
  //         FFTW_BACKWARD, FFTW_ESTIMATE);
  // }
  FFTEngine(cv::Mat& src) {
    int H = src.rows; int W = src.cols; int C = src.channels();
    int n[2] = {H, W};
    p_ = fftwf_plan_many_dft(2, n, C,
          (fftwf_complex*)src.data, n, 1, H*W,
          (fftwf_complex*)src.data, n, 1, H*W,
          FFTW_FORWARD, FFTW_ESTIMATE);
    ip_ = fftwf_plan_many_dft(2, n, C,
          (fftwf_complex*)src.data, n, 1, H*W,
          (fftwf_complex*)src.data, n, 1, H*W,
          FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  ~FFTEngine() {
    fftwf_destroy_plan(p_);
    fftwf_destroy_plan(ip_);
  }

  void fft() {
    fftwf_execute(p_);
  }
  void ifft() {
    fftwf_execute(ip_);
  }

private:
  fftwf_plan p_;
  fftwf_plan ip_;
// protected:
};


#endif
