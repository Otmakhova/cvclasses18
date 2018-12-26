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

void corner_detector_fast::detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray /*mask = cv::noArray()*/)
{
    keypoints.clear();
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

corner_detector_fast::corner_detector_fast()
{
    m_testAreaSize = 31;
    m_testPointsNum = 256;
    m_descriptorBytesNum = m_testPointsNum / 8;
    m_sigma = m_testAreaSize / 5.0;

    generateTestPoints();
}

void corner_detector_fast::generateTestPoints()
{
    cv::RNG rng;
    cv::Point2i point1, point2;

    for (int i = 0; i < m_testPointsNum; i++)
    {
        point1.x = cvRound(rng.gaussian(m_sigma));
        point1.y = cvRound(rng.gaussian(m_sigma));
        point2.x = cvRound(rng.gaussian(m_sigma));
        point2.y = cvRound(rng.gaussian(m_sigma));

        m_testPoints.push_back(std::make_pair(point1, point2));
    }
}

void corner_detector_fast::calcDescriptor(const cv::Point2i& keypoint, cv::Mat& descriptor)
{
    cv::Point2i point1, point2;
    bool testResult;

    for (int byteNum = 0; byteNum < m_descriptorBytesNum; byteNum++)
    {
        descriptor.at<uchar>(byteNum) = 0;
        for (int bitNum = 0; bitNum < 8; bitNum++)
        {
            point1 = keypoint + m_testPoints[byteNum * 8 + bitNum].first;
            point2 = keypoint + m_testPoints[byteNum * 8 + bitNum].second;

            testResult = m_imageForDescriptor.at<uchar>(point1) < m_imageForDescriptor.at<uchar>(point2);
            descriptor.at<uchar>(byteNum) |= (testResult << bitNum);
        }
    }
}

void corner_detector_fast::compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    if (image.empty())
        return;
    else if (image.channels() == 1)
        image.copyTo(m_imageForDescriptor);
    else if (image.channels() == 3)
        cv::cvtColor(image, m_imageForDescriptor, cv::COLOR_BGR2GRAY);

    cv::GaussianBlur(m_imageForDescriptor, m_imageForDescriptor, cv::Size(9, 9), 2.0, 2.0, cv::BORDER_REPLICATE);

    cv::copyMakeBorder(m_imageForDescriptor, m_imageForDescriptor, m_testAreaSize, m_testAreaSize, m_testAreaSize, m_testAreaSize, cv::BORDER_REPLICATE);

    cv::Mat descriptorsMat(keypoints.size(), m_descriptorBytesNum, CV_8U, cv::Scalar(0));

    for (int i = 0; i < keypoints.size(); i++)
        calcDescriptor(cv::Point2i(keypoints[i].pt) + cv::Point2i(m_testAreaSize, m_testAreaSize), descriptorsMat.row(i));

    descriptors.assign(descriptorsMat);
}

void corner_detector_fast::detectAndCompute(cv::InputArray image, cv::InputArray, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors, bool /*= false*/)
{
    threshold = 20;
    detect(image, keypoints);
    compute(image, keypoints, descriptors);
}
} // namespace cvlib
