/* Descriptor matcher algorithm implementation.
 * @file
 * @date 2018-11-25
 * @author Anonymous
 */

#include "cvlib.hpp"

namespace cvlib
{
void descriptor_matcher::knnMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, int k /*unhandled*/,
                                      cv::InputArrayOfArrays masks /*unhandled*/, bool compactResult /*unhandled*/)
{
    if (trainDescCollection.empty())
        return;

    auto q_desc = queryDescriptors.getMat();
    auto& t_desc = trainDescCollection[0];

    matches.resize(q_desc.rows);

    cv::RNG rnd;
    for (int i = 0; i < q_desc.rows; ++i)
    {
        // \todo implement Ratio of SSD check.
        //matches[i].emplace_back(i, rnd.uniform(0, t_desc.rows), FLT_MAX);
    }
}

void descriptor_matcher::radiusMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, float maxDistance,
                                         cv::InputArrayOfArrays masks /*unhandled*/, bool compactResult /*unhandled*/)
{
    if (trainDescCollection.empty())
        return;

    auto q_desc = queryDescriptors.getMat();
    auto& t_desc = trainDescCollection[0];
    matches.resize(q_desc.rows);

    std::vector<int> trueMatches; // good matches for knnMatch
    std::vector<double> distances;

    for (auto i = 0; i < q_desc.rows; ++i)
    {
        for (auto j = 0; j < t_desc.rows; ++j)
        {
            double dist = cv::norm(q_desc.row(i) - t_desc.row(j), cv::NORM_L2);
            if (dist < maxDistance)
            {
                trueMatches.push_back(j);
                distances.push_back(dist);
            }
        }

        int nearestPoint = 0;
        if (!trueMatches.empty())
        {
            int minDistIndex = std::min_element(distances.begin(), distances.end()) - distances.begin();
            double minDist = *std::min_element(distances.begin(), distances.end());

            nearestPoint = trueMatches[minDistIndex];

            bool truePair = true;
            for (auto k = 0; k < trueMatches.size(); k++)
            {
                if (trueMatches.at(k) != nearestPoint && minDist / distances.at(k) > ratio_)
                {
                    truePair = false;
                    break;
                }
            }

            if (truePair)
                matches[i].emplace_back(i, nearestPoint, (float)minDist);
        }
        trueMatches.clear();
        distances.clear();
    }
	// \todo implement matching with "maxDistance"
    //knnMatchImpl(queryDescriptors, matches, 1, masks, compactResult);
}
} // namespace cvlib