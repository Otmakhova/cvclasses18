/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <ctime>

namespace cvlib
{
// static
cv::Ptr<corner_detector_fast> corner_detector_fast::create()
{
    return cv::makePtr<corner_detector_fast>();
}

corner_detector_fast::corner_detector_fast()
{
    double sigma = 1.0 / 25.0 * S * S;
    for (int i = 0; i < all_length; i++)
    {
        test_points1[i] = cv::Point2f(rng.gaussian(sigma), rng.gaussian(sigma));
        test_points2[i] = cv::Point2f(rng.gaussian(sigma), rng.gaussian(sigma));
    }
}

void corner_detector_fast::detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray /*mask = cv::noArray()*/)
{
    keypoints.clear();
    // \todo implement FAST with minimal LOCs(lines of code), but keep code readable.
    num_point = 0;
    cv::Size image_size = image.size();
    cv::Mat imag = image.getMat();

    std::vector<int> etalon_1(12);
    std::vector<int> etalon_2(12);
    std::fill(etalon_1.begin(), etalon_1.end(), 1);
    std::fill(etalon_2.begin(), etalon_2.end(), 2);

    int i_c, i_plus_thresh, i_minus_thresh;

    for (auto i = 3; i < image_size.height - 3; i++)
        for (auto j = 3; j < image_size.width - 3; j++)
        {
            i_c = imag.at<unsigned char>(i, j);
            i_plus_thresh = i_c + threshold;
            i_minus_thresh = i_c - threshold;

            for (auto k = 0; k < 16; k++)
                circle_points.push_back(imag.at<unsigned char>(i + offset_i[k], j + offset_j[k]));

            int light = 0, dark = 0;
            for (auto l = circle_points.begin(); l != circle_points.end(); l += 4)
            {
                light = *l > i_plus_thresh ? ++light : light;
                dark = *l < i_minus_thresh ? ++dark : dark;
            }

            if (light > 2 || dark > 2)
            {
                copyVector();
                std::transform(cyclic_buffer.begin(), cyclic_buffer.end(), cyclic_buffer.begin(),
                               [=](int n) { return (n < i_minus_thresh) + 2 * (n > i_plus_thresh); });

                auto it_dark = std::search(cyclic_buffer.begin(), cyclic_buffer.end(), etalon_1.begin(), etalon_1.end());
                auto it_ligth = std::search(cyclic_buffer.begin(), cyclic_buffer.end(), etalon_2.begin(), etalon_2.end());

                if (it_dark != cyclic_buffer.end() || it_ligth != cyclic_buffer.end())
                {
                    num_point++;
                    keypoints.push_back(cv::KeyPoint(cv::Point2f(double(j), double(i)), 10, float(0)));
                }
                cyclic_buffer.clear();
            }
            circle_points.clear();
        }
}

bool corner_detector_fast::pointOnImage(const cv::Mat& image, const cv::Point2f& point)
{
    if (point.x > 0.0 && point.x < image.rows && point.y > 0.0 && point.y < image.cols)
        return true;
    return false;
}

int corner_detector_fast::twoPointsTest(const cv::Mat& image, const cv::Point2f& point1, const cv::Point2f& point2, const int& num)
{
    if (pointOnImage(image, point1) && pointOnImage(image, point2) && image.at<uchar>(point1) < image.at<uchar>(point2))
    {
        return 1 << num;
    }
    return 0;

}
void corner_detector_fast::binaryTest(const cv::Mat& image, const cv::Point2f& keypoint, int* descriptor)
{
    for (int i = 0; i < all_length; i++)
    {
        descriptor[i / 32] += twoPointsTest(image, keypoint + test_points1[i], keypoint + test_points2[i], i % 32);
    }
}
void corner_detector_fast::compute(cv::InputArray input, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    // \todo implement any binary descriptor
    cv::Mat image;
    cv::GaussianBlur(input.getMat(), image, cv::Size(9, 9), 2.0);

    descriptors.create(static_cast<int>(keypoints.size()), desc_length, CV_32S);
    cv::Mat desc_mat = descriptors.getMat();

    int* desc_ptr = desc_mat.ptr<int>();

    const int keypoints_num = keypoints.size();
    int shift = 0;
    for (int i = 0; i < keypoints_num; i++)
    {
        shift = i * desc_length;
        binaryTest(image, keypoints[i].pt, &desc_ptr[shift]);
    }
}

void corner_detector_fast::detectAndCompute(cv::InputArray image, cv::InputArray, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors, bool /*= false*/)
{
    threshold = 20;
    detect(image, keypoints);
    compute(image, keypoints, descriptors);
}
} // namespace cvlib
