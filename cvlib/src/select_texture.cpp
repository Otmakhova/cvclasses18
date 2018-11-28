/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include "cvlib.hpp"

namespace
{
struct descriptor : public std::vector<double>
{
    using std::vector<double>::vector;
    descriptor operator-(const descriptor& right) const
    {
        descriptor temp = *this;
        for (size_t i = 0; i < temp.size(); ++i)
        {
            temp[i] -= right[i];
        }
        return temp;
    }

    double norm_l1() const
    {
        double res = 0.0;
        for (auto v : *this)
        {
            res += std::abs(v);
        }
        return res;
    }

    double norm_l2() const
    {
        double res = 0.0;
        for (auto v : *this)
        {
            res += v * v;
        }
        return sqrt(res);
    }
};

class TextureSegmentation
{
    public:
    cv::Mat select_texture(const cv::Mat& image, const cv::Rect& roi, double eps)
    {
        int kernel_size = std::min(roi.height, roi.width) / 2;
        if (kernel_size % 2 == 0)
            kernel_size -= 1;
        // create Gabor filters
        filters.clear();
        createFilters(filters, kernel_size);
        // eval convolution
		std::vector<cv::Mat> intConv(filters.size());
        std::vector<cv::Mat> intSqConv(filters.size());
        evalFilters(image, filters, intConv, intSqConv);

        descriptor reference;
        calculateDescriptor(roi, reference, intConv, intSqConv);

        cv::Mat res = cv::Mat::zeros(image.size(), CV_8UC1);

        descriptor test(reference.size());
        cv::Rect baseROI = roi - roi.tl();

        int last_i = image.size().width - baseROI.width;
        int last_j = image.size().height - baseROI.height;

        for (int i = 0; i < last_i; i ++)
        {
            for (int j = 0; j < last_j; j ++)
            {
                auto curROI = baseROI + cv::Point(i, j);
                calculateDescriptor(curROI, test, intConv, intSqConv);
                if ((test - reference).norm_l2() <= eps)
                    res(curROI) = 255;
            }
        }

        return res;
    }

    private:
    void evalFilters(const cv::Mat& image, const std::vector<cv::Mat>& filters, std::vector<cv::Mat>& intConv, std::vector<cv::Mat>& intSqConv)
    {
        cv::Mat filter_res;
        for (size_t i = 0; i < filters.size(); i++)
        {
            filter_res.release();
            cv::filter2D(image, filter_res, CV_32F, filters[i]);
            cv::integral(filter_res, intConv[i], intSqConv[i]);
        }
    }

    void calculateDescriptor(const cv::Rect& rect, descriptor& descr, const std::vector<cv::Mat>& intConv, const std::vector<cv::Mat>& intSqConv)
    {
        descr.clear();
        int x1 = rect.x;
        int x2 = rect.x + rect.width;
        int y1 = rect.y;
        int y2 = rect.y + rect.height;
        for (size_t i = 0; i < intConv.size(); i++)
        {
            double S1 = intConv[i].at<double>(y2, x2) + intConv[i].at<double>(y1, x1) - intConv[i].at<double>(y2, x1) - intConv[i].at<double>(y1, x2);
            double S2 =
                intSqConv[i].at<double>(y2, x2) + intSqConv[i].at<double>(y1, x1) - intSqConv[i].at<double>(y2, x1) - intSqConv[i].at<double>(y1, x2);
            double mean = double(S1) / (x2 - x1) / (y2 - y1);
            double std = sqrt(1.0 / (x2 - x1) / (y2 - y1) * (S2 - 2 * mean * S1) + mean * mean);
            descr.emplace_back(mean);
            descr.emplace_back(std);
        }
    }

    void createFilters(std::vector<cv::Mat>& filters, const int kernel_size)
    {
        filters.clear();
        const double th = CV_PI / 4;
        const double lm = 10.0;
        const double gm = 0.5;

        for (auto sig = 5; sig <= 15; sig += 5)
        {
            filters.emplace_back(cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sig, th, lm, gm));
            filters.back().convertTo(filters.back(), CV_32F);
        }
    }

    cv::Mat image;
    std::vector<cv::Mat> filters;
};
} // namespace

namespace cvlib
{
TextureSegmentation texture;
cv::Mat select_texture(const cv::Mat& image, const cv::Rect& roi, double eps)
{
    cv::Mat res = texture.select_texture(image, roi, eps);
    return res;
}
} // namespace cvlib
