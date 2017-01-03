/*!
 * \file
 * \brief
 * \author Aleksandra Karbarczyk
 */

#include <memory>
#include <string>

#include "CvSolvePnPRansac.hpp"
#include "Common/Logger.hpp"
#include "Types/HomogMatrix.hpp"

#include <boost/bind.hpp>

#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;
using namespace boost;

namespace Processors {
namespace CvSolvePnPRansac {

CvSolvePnPRansac::CvSolvePnPRansac(const std::string & name):
        Base::Component(name),
		iterations_count("iterationsCount", 100),
		reprojection_error("reprojectionError", 8.0),
		confidence("confidence", 100) {
//		flag("flag", "ITERATIVE", std::string("flag")) {
	registerProperty(iterations_count);
	registerProperty(reprojection_error);
	registerProperty(confidence);
//	registerProperty(flag);
}

CvSolvePnPRansac::~CvSolvePnPRansac() {
}

void CvSolvePnPRansac::prepareInterface() {
	// Register data streams, events and event handlers HERE!
	registerStream("in_object3d", &in_object3d);
	registerStream("in_camera_info", &in_camera_info);
    registerStream("in_homog_matrix", &in_homog_matrix);
    registerStream("in_rvec", &in_rvec);
    registerStream("in_tvec", &in_tvec);

	registerStream("out_homog_matrix", &out_homog_matrix);
	registerStream("out_rvec", &out_rvec);
	registerStream("out_tvec", &out_tvec);
    registerStream("out_inliers", &out_inliers);

	// Register handlers
	registerHandler("onNewObject3D", boost::bind(&CvSolvePnPRansac::onNewObject3D, this));
	addDependency("onNewObject3D", &in_object3d);
	addDependency("onNewObject3D", &in_camera_info);

}

bool CvSolvePnPRansac::onInit() {
	return true;
}

bool CvSolvePnPRansac::onFinish() {
	return true;
}

bool CvSolvePnPRansac::onStop() {
	return true;
}

bool CvSolvePnPRansac::onStart() {
	return true;
}

void CvSolvePnPRansac::onNewObject3D() {
    boost::shared_ptr<Types::Objects3D::Object3D> object3D = in_object3d.read();
    Types::CameraInfo camera_info = in_camera_info.read();

    Mat modelPoints(object3D->getModelPoints());
    Mat imagePoints(object3D->getImagePoints());

    Mat_<double> rvec(3, 1);
    Mat_<double> tvec(3, 1);
    Mat_<double> rotation_matrix;

    vector<int> inliers;
    bool use_extrinsic_guess = true;

    if (!in_rvec.empty() && !in_tvec.empty()) {
        rvec = in_rvec.read();
        tvec = in_tvec.read();
    } else if (!in_homog_matrix.empty()) {
        Types::HomogMatrix homog_matrix = in_homog_matrix.read();
        Mat_<double> temp_rotation_matrix;
        temp_rotation_matrix.create(3, 3);

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                temp_rotation_matrix(i, j) = homog_matrix(i, j);
            }
            tvec(i, 0) = homog_matrix(i, 3);
        }
        Rodrigues(temp_rotation_matrix, rvec);
    } else {
        use_extrinsic_guess = false;
    }

    solvePnPRansac(modelPoints, imagePoints, camera_info.cameraMatrix(), camera_info.distCoeffs(), rvec, tvec, use_extrinsic_guess,
				   iterations_count, reprojection_error, confidence, inliers);
    Rodrigues(rvec, rotation_matrix);

    CLOG(LERROR) << "**** Inliers number: " << inliers.size();
    CLOG(LINFO) << "rvec = " << rvec << "  tvec = " << tvec;

    // Create homogenous matrix.
    cv::Mat pattern_pose = (cv::Mat_<double>(4, 4) <<
            rotation_matrix(0, 0), rotation_matrix(0, 1), rotation_matrix(0, 2), tvec(0),
            rotation_matrix(1, 0), rotation_matrix(1, 1), rotation_matrix(1, 2), tvec(1),
            rotation_matrix(2, 0), rotation_matrix(2, 1), rotation_matrix(2, 2), tvec(2),
            0, 0, 0, 1);

    CLOG(LINFO) << "pattern_pose:\n" << pattern_pose;

    out_rvec.write(rvec.clone());
    out_tvec.write(tvec.clone());
    out_homog_matrix.write(Types::HomogMatrix(pattern_pose.clone()));
    out_inliers.write(inliers);
}



} //: namespace CvSolvePnPRansac
} //: namespace Processors
